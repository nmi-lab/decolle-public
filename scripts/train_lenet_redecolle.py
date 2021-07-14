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
from decolle.lenet_redecolle import RecLIFLayer, LenetREDECOLLE
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
from pylab import plt

np.set_printoptions(precision=4)
args = parse_args('parameters/params.yml')
device = args.device


starting_epoch = 0

params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']

dataset = importlib.import_module(params['dataset'])
try:
    create_data = dataset.create_data
except AttributeError:
    create_data = dataset.create_dataloader

verbose = args.verbose

## Load Data
gen_train, gen_test = create_data(chunk_size_train=params['chunk_size_train'],
                                  chunk_size_test=params['chunk_size_test'],
                                  batch_size=params['batch_size'],
                                  dt=params['deltat'],
                                  num_workers=params['num_dl_workers'])

data_batch, target_batch = next(iter(gen_train))
data_batch = data_batch.to(torch.float32).to(device)
target_batch = target_batch.to(torch.float32).to(device)

#d, t = next(iter(gen_train))
input_shape = data_batch.shape[-3:]

#Backward compatibility
if 'dropout' not in params.keys():
    params['dropout'] = [.5]

## Create Model, Optimizer and Loss
net = LenetREDECOLLE( out_channels=params['out_channels'],
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
                    lif_layer_type = [LIFLayer]*len(params['Nhid'])+[RecLIFLayer]*len(params['Mhid']),
                    method=params['learning_method'],
                    with_output_layer=params['with_output_layer']).to(device)


opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])

reg_l = params['reg_l'] if 'reg_l' in params else None

if params['loss_scope']=='global':
    loss = [None for i in range(len(net))]
    loss[-1] = cross_entropy_one_hot
    decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)
else:
    loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
    if net.with_output_layer:
        loss[-1] = cross_entropy_one_hot
    decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)

##Initialize
net.init_parameters(data_batch)

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

def process_decolle_output(net, data_batch):
    from decolle.utils import tonp
    net.init(data_batch, burnin=1)
    t = (data_batch.shape[1],)
    s,r,u = net(data_batch[:,0])
    s_out = [np.zeros(t+tonp(layer).shape     ) for layer in s]
    r_out = [np.zeros(t+tonp(layer).shape     ) for layer in r ]
    u_out = [np.zeros(t+tonp(layer).shape     ) for layer in u]


    for t in range(data_batch.shape[1]):
        net.state=None
        s,r,u = net(data_batch[:,t])
        for i in range(len(net.LIF_layers)):
            s_out[i][t,:] = tonp(s[i])
            u_out[i][t,:] = tonp(u[i])
            r_out[i][t,:] = tonp(r[i])

    return s_out, r_out, u_out

def show_uhistograms(net, data_batch):
    s_out, r_out, u_out =  process_decolle_output(net, data_batch)
    rng = np.arange(-10,10,1)
    plt.figure()
    plt.imshow(np.array([np.histogram(u_out[0][t:t+10,:,0].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[-100,100,0,300])
    plt.figure()
    plt.imshow(np.array([np.histogram(u_out[1][t:t+10,:,0].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[-100,100,0,300])
    plt.figure()
    plt.imshow(np.array([np.histogram(u_out[2][t:t+10,:,0].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[-100,100,0,300])

    plt.show()

def show_rhistograms(net, data_batch):
    s_out, r_out, u_out =  process_decolle_output(net, data_batch)
    rng = np.arange(-1,1,.1)
    plt.figure()
    plt.imshow(np.array([np.histogram(r_out[0][t:t+10,:].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[min(rng),max(rng),0,1])
    plt.figure()
    plt.imshow(np.array([np.histogram(r_out[1][t:t+10,:].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[min(rng),max(rng),0,1])
    plt.figure()
    plt.imshow(np.array([np.histogram(r_out[2][t:t+10,:].reshape(-1), bins=rng)[0] for t in range(10,300,10)]), extent=[min(rng),max(rng),0,1])

    plt.show()



# --------TRAINING LOOP----------
if not args.no_train:
    test_acc_hist = []
    for e in range(starting_epoch , params['num_epochs'] ):
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

            test_loss, test_acc = test(gen_test, decolle_loss, net, params['burnin_steps'], print_error = True)
            test_acc_hist.append(test_acc)

            if not args.no_save:
                write_stats(e, test_acc, test_loss, writer)
                np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)

        total_loss, act_rate = train(gen_train, decolle_loss, net, opt, e, params['burnin_steps'], online_update=params['online_update'])
        if not args.no_save:
            for i in range(len(net)):
                writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)



