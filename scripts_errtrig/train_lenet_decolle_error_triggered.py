#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_lenet_decolle_error_triggered
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from decolle.lenet_decolle_model_errortriggered import LenetDECOLLEErrorTriggered, DECOLLELoss, LIFLayerVariableTau, LIFLayer, BoxLIFLayer
from decolle.experimental.boxlif import *
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot, tonp
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib

def train(gen_train, decolle_loss, net, opt, epoch, burnin, online_update=True, batches_per_epoch=-1):
    '''
    Trains a DECOLLE network

    Arguments:
    gen_train: a dataloader
    decolle_loss: a DECOLLE loss function, as defined in base_model
    net: DECOLLE network
    opt: optimizaer
    epoch: epoch number, for printing purposes only
    burnin: time during which the dynamics will be run, but no updates are made
    online_update: whether updates should be made at every timestep or at the end of the sequence.
    '''
    device = net.get_input_layer_device()
    iter_gen_train = iter(gen_train)
    total_loss = np.zeros(decolle_loss.num_losses)
    act_rate = [0 for i in range(len(net))]

        
    loss_tv = torch.tensor(0.).to(device)
    net.train()
    dtype = net.LIF_layers[0].base_layer.weight.dtype  
    batch_iter = 0
    
    error_rates = []
    for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(epoch)):
        data_batch = torch.Tensor(data_batch).type(dtype).to(device)
        target_batch = torch.Tensor(target_batch).type(dtype).to(device)
        if len(target_batch.shape) == 2:
            #print('replicate targets for all timesteps')
            target_batch = target_batch.unsqueeze(1)
            shape_with_time = np.array(target_batch.shape)
            shape_with_time[1] = data_batch.shape[1]
            target_batch = target_batch.expand(*shape_with_time)

        loss_mask = (target_batch.sum(2)>0).unsqueeze(2).float()
        # loss_mask = (data_batch.reshape(data_batch.shape[0],data_batch.shape[1],-1).mean(2)>0.01).unsqueeze(2).float()
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
        
        net.update_thetas()
        error_rates .append(np.array([net.err_enc_layers[i].last_err_rate for i in range(len(net))]))

    total_loss /= t_sample
    print('Loss {0}'.format(total_loss))
    print('Activity Rate {0}'.format(act_rate))
    return total_loss, act_rate, error_rates

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
                                  num_workers=params['num_dl_workers'],
                                  drop_last=True)

data_batch, target_batch = next(iter(gen_train))
data_batch = torch.Tensor(data_batch).to(device)
target_batch = torch.Tensor(target_batch).to(device)

#d, t = next(iter(gen_train))
input_shape = data_batch.shape[-3:]

if not hasattr(args, 'psp'):
    args.psp = 'BoxLIFLayer'

if args.psp == 'nobox' or args.psp == 'LIFLayer':
    neuron_model = LIFLayer
elif args.psp == 'box' or args.psp == 'BoxLIFLayer':
    neuron_model = BoxLIFLayer
print(neuron_model)

## Create Model, Optimizer and Loss
net = LenetDECOLLEErrorTriggered(
                    out_channels=params['out_channels'],
                    init_theta = params['init_theta'],
                    Nhid=params['Nhid'],
                    Mhid=params['Mhid'],
                    kernel_size=params['kernel_size'],
                    pool_size=params['pool_size'],
                    input_shape=params['input_shape'],
                    alpha=params['alpha'],
                    alpharp=params['alpharp'],
                    beta=params['beta'],
                    num_conv_layers=params['num_conv_layers'],
                    num_mlp_layers=params['num_mlp_layers'],
                    lc_ampl=params['lc_ampl'],
                    lif_layer_type = neuron_model,
                    method=params['learning_method'],
                    set_point_err=params['error_rate'],
                    with_output_layer=False).to(device)

if hasattr(params['learning_rate'], '__len__'):
    from decolle.utils import MultiOpt
    opts = []
    for i in range(len(params['learning_rate'])):
        if params['optimizer'] == 'adamax':
            opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
        elif params['optimizer'] == 'sgd':
            opts.append(torch.optim.SGD(net.get_trainable_parameters(i), lr=params['learning_rate'][i]))
        else:
            raise NotImplementedError(params['optimizer'] + ' optimizer is not supported')
    opt = MultiOpt(*opts)
else:
    if params['optimizer'] == 'adamax':
        opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    elif params['optimizer'] == 'sgd':
        opt = torch.optim.SGD(net.get_trainable_parameters(), lr=params['learning_rate'])
    else:
        raise NotImplementedError(params['optimizer'] + ' optimizer is not supported')
print(opt)

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

# --------TRAINING LOOP----------
if not args.no_train:
    test_acc_hist = []
    error_rate_hist = []
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

        total_loss, act_rate, error_rates = train(gen_train, decolle_loss, net, opt, e, params['burnin_steps'], online_update=params['online_update'], batches_per_epoch = params['batches_per_epoch'])
        
        
        error_rate_hist.append(error_rates)
        if not args.no_save:
            np.save(log_dir+'/error_rate.npy', np.array(error_rate_hist),)
        
        print(net.err_enc_layers)
        if not args.no_save:
            for i in range(len(net)):
                writer.add_scalar('/error_rate/{0}'.format(i), net.err_enc_layers[i].last_err_rate, e)
                writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
