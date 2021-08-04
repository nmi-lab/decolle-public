# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:47:55 2021

@author: stanzarella
"""
#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_lenet_decolle
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
from decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayerVariableTau, LIFLayer
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot, tonp, prediction_mostcommon
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib

sigmoid = torch.sigmoid


'''
COMMENTS by Simone (1)

In the first to sections (Parameters LENET and Instanciate LENET) I copied the 
same lines of code of the script "train_lenet_decolle.py" 

However, I load 'parameters/paramsTorchVis28.yml' which contains different 
parameters in ordet to load a MNIST with figure 28x28. This MNIST is loaded and
spike-encoded in section 3 as in the Zenke's notebooks 
(spytorch/SpyTorchTutorial2.ipynb at master · fzenke/spytorch · GitHub)
'''


## *1. SET Parameters LENET --------------------------------------------------*

np.set_printoptions(precision=4)
args = parse_args('parameters/paramsTorchVis28.yml')
device = args.device


starting_epoch = 0

params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']

if 'dropout' not in params.keys():
    params['dropout'] = [.5]
    
## *2. Instanciate LENET    --------------------------------------------------*
    
net = LenetDECOLLE( out_channels=params['out_channels'],
                    Nhid=params['Nhid'],
                    Mhid=params['Mhid'],
                    kernel_size=params['kernel_size'],
                    pool_size=params['pool_size'],
                    input_shape=params['input_shape'],
                    alpha=params['alpha'],
                    alpharp=params['alpharp'],
                    dropout=params['dropout'],
                    beta=params['beta'],
                    num_conv_layers=params['num_conv_layers'],
                    num_mlp_layers=params['num_mlp_layers'],
                    lc_ampl=params['lc_ampl'],
                    lif_layer_type = LIFLayer,
                    method=params['learning_method'],
                    with_output_layer=True).to(device)

if hasattr(params['learning_rate'], '__len__'):
    from decolle.utils import MultiOpt
    opts = []
    for i in range(len(params['learning_rate'])):
        opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
    opt = MultiOpt(*opts)
else:
    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])

reg_l = params['reg_l'] if 'reg_l' in params else None

if 'loss_scope' in params and params['loss_scope']=='crbp':
    from decolle.lenet_decolle_model import CRBPLoss
    loss = torch.nn.SmoothL1Loss(reduction='none')
    decolle_loss = CRBPLoss(net = net, loss_fn = loss, reg_l=reg_l)
else:
    loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
    if net.with_output_layer:
        loss[-1] = cross_entropy_one_hot
    decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)
    
## * 3. Load DATA (Zenke's spike-encoded MNIST)-----------------------------------------------------*

'''
COMMENTS by Simone (2)

In section 3 I use the functions "current2firing_time", "sparse_data_generator",
"sparse_data_generator_1batch" for the spike encoding of a MNIST.
The MNIST is the TorchVision FashonMNIST. As the MNIST used in DECOLLE,
it is divided in mini-batch (slices).
Each batch contains 256 28x28 figures. The idea is encoding the pixel-intensity 
information with a spiking information, and it is done in "current2firing_time".
In Kaiser 2010 the spiking MNIST was obtained by a "Dynamic Vision
Sensor (DVS) mounted on a pantilt unit performing microsaccadic motions 
in front of a screen displaying samples from the MNIST dataset".
" The samples are cropped spatially from 34 × 34 to 32 × 32 and temporally to 300 ms for
both the train and test set."
" A N-MNIST sample is therefore represented as a tensor of shape 300 × 2 × 32 × 32, 
stacked into mini-batch of 500 samples. "

So, here we created a spiking-MNIST with dimension
[num_mini_batch, 256, 300, 1, 28, 28],
instead of the Kaiser's dimension
[num_mini_batch, 500, 300, 1, 32, 32],

QUESTION : is the Kaiser MNIST supposed to be binary, a sparse tensor composed by 1 and 0?

Because here we create actually a tensor (in "sparse_data_generator" function)
in the way:
    
X_batch = ...
torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units[0],nb_units[1],nb_units[2]])).to(device)

FOR EACH BATCH 

WHERE 
- i is the 5-axes domain of the tensor and 
- v is a ones-array to set to 1 the sparse tensor in the coordinates prescribed by i.

'''


import torchvision
import matplotlib.pyplot as plt

# The coarse network structure is dedicated to the Fashion MNIST dataset. 
# nb_inputs  = [28,28]

time_step = 1e-3
nb_steps  = params['chunk_size_train']

batch_size = params['batch_size']

isInitializated = False

# a. Functions to load pre-process
def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    T[idx] = tmax
    return T
 

def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True ):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y,dtype=np.float)
    number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3/time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.float)
    unit_numbers = np.arange(nb_units[1]*nb_units[2])
    # unit_numbers_2 = np.arange(nb_units[2])

    if shuffle:
        np.random.shuffle(sample_index)

    # total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(5) ]
        for bc,idx in enumerate(batch_index):
            for n_mat in range(0,nb_units[0]):
                c = firing_times[idx]<nb_steps
                times, units = firing_times[idx][c], unit_numbers[c]
                
                cols = np.fix((units+1)/nb_units[1]-1).astype(int) #cols
                rows = (np.mod((units),nb_units[1])).astype(int) #rows
    
                batch = [bc for _ in range(len(times))]
                matrices = [n_mat for _ in range(len(times))]
                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(matrices)
                coo[3].extend(rows)
                coo[4].extend(cols)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units[0],nb_units[1],nb_units[2]])).to(device)
        # y_batch = torch.tensor(labels_[batch_index],device=device)
        LL = []
        for lab in labels_[batch_index] :
            LL.append([[lab]])
        y_batch = torch.tensor(np.repeat( np.array(LL) , 300, axis=1 ),device=device)

        yield X_batch.detach().cpu().to_dense().numpy(), y_batch.detach().cpu().numpy()

        counter += 1
        
def sparse_data_generator_1batch(X, y, batch_size, nb_steps, nb_units, nbatch = 0, shuffle=True ):
    
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y,dtype=np.float)
    # number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3/time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.float)
    unit_numbers = np.arange(nb_units[1]*nb_units[2])

    if shuffle:
        np.random.shuffle(sample_index)

    # total_batch_count = 0
    counter = 0
    counter = nbatch
    batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

    coo = [ [] for i in range(5) ]
    for bc,idx in enumerate(batch_index):
        for n_mat in range(0,nb_units[0]):
            c = firing_times[idx]<nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]
            
            cols = np.fix((units+1)/nb_units[1]-1).astype(int) #cols
            rows = (np.mod((units),nb_units[1])).astype(int) #rows

            batch = [bc for _ in range(len(times))]
            matrices = [n_mat for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(matrices)
            coo[3].extend(rows)
            coo[4].extend(cols)

    i = torch.LongTensor(coo).to(device)
    v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

    X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units[0],nb_units[1],nb_units[2]])).to(device)
    y_batch = torch.tensor(labels_[batch_index],device=device)

    return X_batch, y_batch

        
# b. LOAD DATA, standardiazation (0,1), and prepare test, train set and labels
    
# Here we load the Dataset
root = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None, download=True)


# Standardize data : from [0,255] to [0,1]
# x_train = torch.tensor(train_dataset.train_data, device=device, dtype=dtype)
x_train = np.array(train_dataset.data, dtype=np.float)
x_train = x_train.reshape(x_train.shape[0],-1)/255
# x_test = torch.tensor(test_dataset.test_data, device=device, dtype=dtype)
x_test = np.array(test_dataset.data, dtype=np.float)
x_test = x_test.reshape(x_test.shape[0],-1)/255

# y_train = torch.tensor(train_dataset.train_labels, device=device, dtype=dtype)
# y_test  = torch.tensor(test_dataset.test_labels, device=device, dtype=dtype)
y_train = np.array(train_dataset.targets, dtype=np.float)
y_test  = np.array(test_dataset.targets, dtype=np.float)

# Here we plot one of the raw data points as an example
data_id = 1
plt.imshow(x_train[data_id].reshape(28,28), cmap=plt.cm.gray_r)
plt.axis("off")

## ***********+---------------------------------------------------------------*
## 4. TRAINING ---------------------------------------------------------------*
## ***********+---------------------------------------------------------------*

'''
COMMENTS by Simone (3)

Here I copied and modified the two functions, train and test, 
of the file "utils.py" of the DECOLLE library.

Now they are re-named "trainTorchVis28" and "testTorchVis28" respectively.

The MAIN CHANGE is the substitution of 
" for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(epoch)): " 

with

" for data_batch, target_batch in sparse_data_generator(x_data, y_data, batch_size, nb_steps, params['input_shape']): "

Finally, also the TRAINING LOOP is the same but I inserted inside the initialization
at the first iteration, with :
    
if not(isInitializated) :
    # Initialize
    data_batch = sparse_data_generator_1batch(x_train, y_train, batch_size, nb_steps, params['input_shape'], nbatch = 10)

    data_batch_tens = np.array(data_batch)[0].to_dense()

    net.init_parameters(data_batch_tens)
    isInitializated = True
    
I could eventually move it outside, as in the DECOLLE code, 
but actually it is called before the first training iteration, so it's exactly
the same.

'''

# a. Training function intra-epoch

def trainTorchVis28(x_data, y_data, decolle_loss, net, opt, epoch, burnin, online_update=True, batches_per_epoch=-1):
    '''
    Trains a DECOLLE network

    Arguments:
        
    SUBSTITUTE (gen_train: a dataloader) WITH -> x_data, y_data
    
    decolle_loss: a DECOLLE loss function, as defined in base_model
    net: DECOLLE network
    opt: optimizaer
    epoch: epoch number, for printing purposes only
    burnin: time during which the dynamics will be run, but no updates are made
    online_update: whether updates should be made at every timestep or at the end of the sequence.
    '''
    device = net.get_input_layer_device()
    # iter_gen_train = iter(gen_train)
    total_loss = np.zeros(decolle_loss.num_losses)
    act_rate = [0 for i in range(len(net))]

        
    loss_tv = torch.tensor(0.).to(device)
    net.train()
    if hasattr(net.LIF_layers[0], 'base_layer'):
        dtype = net.LIF_layers[0].base_layer.weight.dtype
    else:
        dtype = net.LIF_layers[0].weight.dtype
    batch_iter = 0
    
    # for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(epoch)):
    for data_batch, target_batch in sparse_data_generator(x_data, y_data, batch_size, nb_steps, params['input_shape']):
        
        print('Epoch ' + str(epoch) + ', Batch ' + str(batch_iter))
        
        data_batch = torch.Tensor(data_batch).type(dtype).to(device)
        target_batch = torch.Tensor(target_batch).type(dtype).to(device)
        # data_batch = np.array(data_batch)[0]
        # target_batch = np.array(target_batch)[0]
        if len(target_batch.shape) == 2:
            #print('replicate targets for all timesteps')
            target_batch = target_batch.unsqueeze(1)
            shape_with_time = np.array(target_batch.shape)
            shape_with_time[1] = data_batch.shape[1]
            target_batch = target_batch.expand(*shape_with_time)

        # loss_mask = (target_batch.sum(2)>0).unsqueeze(2).float()
        loss_mask = (data_batch.reshape(data_batch.shape[0],data_batch.shape[1],-1).mean(2)>0.01).unsqueeze(2).float()
        net.init(data_batch, burnin)
        t_sample = data_batch.shape[1]
        for k in (range(burnin,t_sample)):
            s, r, u = net.forward(data_batch[:, k, :, :])
            loss_ = decolle_loss(s, r, u, target=target_batch[:,k,:], mask = loss_mask[:,k,:], sum_ = False)
            total_loss += tonp(torch.Tensor(loss_))
            loss_tv += sum(loss_)
            if online_update: 
                loss_tv.backward()
                opt.step()
                opt.zero_grad()
                for i in range(len(net)):
                    act_rate[i] += tonp(s[i].mean().data)/t_sample
                loss_tv = torch.tensor(0.).to(device)
        print( 'Loss : ' + str(total_loss))
        if not online_update:
            loss_tv.backward()
            opt.step()
            opt.zero_grad()
            for i in range(len(net)):
                act_rate[i] += tonp(s[i].mean().data)/t_sample
            loss_tv = torch.tensor(0.).to(device)
        batch_iter +=1
        if batches_per_epoch>0:
            if batch_iter >= batches_per_epoch: break

    total_loss /= t_sample
    print('Loss {0}'.format(total_loss))
    print('Activity Rate {0}'.format(act_rate))
    return total_loss, act_rate

## b. TEST FUNCTION

def testTorchVis28(x_data, y_data, decolle_loss, net, burnin, print_error = True, debug = False):
    net.eval()
    if hasattr(net.LIF_layers[0], 'base_layer'):
        dtype = net.LIF_layers[0].base_layer.weight.dtype
    else:
        dtype = net.LIF_layers[0].weight.dtype
    with torch.no_grad():
        device = net.get_input_layer_device()
        # iter_data_labels = iter(gen_test)
        test_res = []
        test_labels = []
        test_loss = np.zeros([decolle_loss.num_losses])

        # for data_batch, target_batch in tqdm.tqdm(iter_data_labels, desc='Testing'):
        for data_batch, target_batch in sparse_data_generator(x_data, y_data, batch_size, nb_steps, params['input_shape']):
            data_batch = torch.Tensor(data_batch).type(dtype).to(device)
            target_batch = torch.Tensor(target_batch).type(dtype).to(device)

            # batch_size = data_batch.shape[0]
            timesteps = data_batch.shape[1]
            nclasses = target_batch.shape[2]
            r_cum = np.zeros((len(net), timesteps-burnin, batch_size, nclasses))



            ## Experimented with target_masking
            #target_mask = (data_batch.mean(2)>0.01).unsqueeze(2).float()
            #target_mask = tonp(data_batch.mean(2, keepdim=True)>0.01).squeeze().unsqueeze(1)
            #target_mask = tonp(data_batch.mean(2, keepdim=True).squeeze().unsqueeze(2)>.01
            net.init(data_batch, burnin)

            for k in (range(burnin,timesteps)):
                s, r, u = net.forward(data_batch[:, k, :, :])
                test_loss_tv = decolle_loss(s,r,u, target=target_batch[:,k], sum_ = False)
                test_loss += [tonp(x) for x in test_loss_tv]
                for n in range(len(net)):
                    r_cum[n,k-burnin,:,:] += tonp(sigmoid(r[n]))
            test_res.append(prediction_mostcommon(r_cum))
            test_labels += tonp(target_batch).sum(1).argmax(axis=-1).tolist()
        test_acc  = accuracy(np.column_stack(test_res), np.column_stack(test_labels))
        test_loss /= np.shape(data_batch)[0]
        if print_error:
            print(' '.join(['Error Rate L{0} {1:1.3}'.format(j, 1-v) for j, v in enumerate(test_acc)]))
    if debug:
        return test_loss, test_acc, s, r, u
    else:
        return test_loss, test_acc
    

## c. WHAT THE ... ? TRY TO DEBUG ...
##Resume if necessary
if args.resume_from is not None:
    print("Checkpoint directory " + checkpoint_dir)
    if not os.path.exists(checkpoint_dir) and not args.no_save:
        os.makedirs(checkpoint_dir)
    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

# Printing parameters
if args.verbose:
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

## d. --------TRAINING LOOP----------

if not args.no_train:
    test_acc_hist = []
    for e in range(starting_epoch , params['num_epochs'] ):
        ## FOR EACH EPOCH
        interval = e // params['lr_drop_interval']
        lr = opt.param_groups[-1]['lr']
        if interval > 0:
            print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
            opt.param_groups[-1]['lr'] = np.array(params['learning_rate']) / (interval * params['lr_drop_factor'])
        else:
            print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
            opt.param_groups[-1]['lr'] = np.array(params['learning_rate'])

        if (e % params['test_interval']) == 0 and e!=0:
            print('---------------Epoch {}-------------'.format(e))
            if not args.no_save:
                print('---------Saving checkpoint---------')
                save_checkpoint(e, checkpoint_dir, net, opt)
            
            # ---------------------------------------------------------------**
            # CALL TEST FUNCTION
            test_loss, test_acc = testTorchVis28(x_train, y_train, decolle_loss, net, params['burnin_steps'], print_error = True)
            # ---------------------------------------------------------------**
            
            test_acc_hist.append(test_acc)

            if not args.no_save:
                write_stats(e, test_acc, test_loss, writer)
                np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)
                
        if not(isInitializated) :
            # Initialize
            data_batch = sparse_data_generator_1batch(x_train, y_train, batch_size, nb_steps, params['input_shape'], nbatch = 10)

            data_batch_tens = np.array(data_batch)[0].to_dense()

            net.init_parameters(data_batch_tens)
            isInitializated = True
            
        # -------------------------------------------------------------------**
            # CALL TEST FUNCTION
        total_loss, act_rate = trainTorchVis28(x_train, y_train, decolle_loss, net, opt, e, params['burnin_steps'], online_update=params['online_update'])
        # -------------------------------------------------------------------**
        
        if not args.no_save:
            for i in range(len(net)):
                writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)