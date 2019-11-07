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
from decolle.lenet_decolle_model import LenetDECOLLE
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
args = parse_args('parameters/params.yml')
device = args.device


starting_epoch = 0

params, writer, dirs = prepare_experiment(name='lenet_decolle_{0}'.format(device), args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']

dataset = importlib.import_module(params['dataset'])
create_dataloader = dataset.create_dataloader


verbose = args.verbose

## Load Data
gen_train, gen_test = create_dataloader(chunk_size_train=params['chunk_size_train'],
                                  chunk_size_test=params['chunk_size_test'],
                                  batch_size=params['batch_size'],
                                  size=params['input_shape'],
                                  dt=params['deltat'], num_workers=params['num_dl_workers'])

data_batch, target_batch = next(iter(gen_train))
data_batch = torch.Tensor(data_batch).to(device)
target_batch = torch.Tensor(target_batch).to(device)

input_shape = data_batch.shape[-3:]

## Create Model, Optimizer and Loss
net = LenetDECOLLE( out_channels=params['out_channels'],
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
                    lc_ampl=params['lc_ampl']).to(device)

if hasattr(params['learning_rate'], '__len__'):
    from decolle.utils import MultiOpt
    opts = []
    for i in range(len(params['learning_rate'])):
        opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
    opt = MultiOpt(*opts)
else:
    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
loss = torch.nn.SmoothL1Loss()

##Resume if necessary
if args.resume_from is not None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

##Initialize
net.init_parameters(data_batch)

# Printing parameters
if args.verbose:
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

# --------TRAINING LOOP----------
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

        test_loss, test_acc = test(gen_test, loss, net, params['burnin_steps'], print_error = True)
        test_acc_hist.append(test_acc)
        
        if not args.no_save:
            write_stats(e, test_acc, test_loss, writer)
        

        np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)
        
    total_loss, act_rate = train(gen_train, loss, net, opt, e, params['burnin_steps'], params['reg_l'])
    for i in range(len(net)):
        writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)




