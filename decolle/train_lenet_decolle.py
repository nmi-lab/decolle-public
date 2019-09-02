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
from lenet_decolle_model import LenetDECOLLE
from load_dvsgestures_sparse import create_data
from utils import parse_args, SummaryWriter, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats
import datetime, os, socket, tqdm
import numpy as np
import torch

args = parse_args('parameters/params.yml')
device = args.device

starting_epoch = 0

params, writer, dirs = prepare_experiment(name='lenet_decolle_{0}'.format(device), args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']

verbose = args.verbose

## Load Data
gen_train, gen_test = create_data(chunk_size_train=params['chunk_size_train'],
                                  chunk_size_test=params['chunk_size_test'],
                                  batch_size=params['batch_size'],
                                  size=params['input_shape'],
                                  dt=params['deltat'])
d, t = gen_train.next()
input_shape = d.shape[-3:]

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
                    random_tau=params['random_tau'],
                    lc_ampl=params['lc_ampl']).to(device)

opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
loss = torch.nn.SmoothL1Loss()

##Resume if necessary
if args.resume_from is not None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    net.init((d,t), 5)
    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

##Initialize
net.init_parameters(d)

# Printing parameters
if verbose:
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

# --------TRAINING LOOP----------
for e in range(starting_epoch , params['num_epochs'] ):
    interval = e // params['lr_drop_interval']
    lr = opt.param_groups[-1]['lr']
    if interval > 0:
        opt.param_groups[-1]['lr'] = params['learning_rate'] / (interval * params['lr_drop_factor'])
    else:
        opt.param_groups[-1]['lr'] = params['learning_rate']
    if lr != opt.param_groups[-1]['lr']:
        print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))

    if e == 1 or (e % params['test_interval']) == 0 and e!=0:
        print('---------------Epoch {}-------------'.format(e))
        if not args.no_save:
            print('---------Saving checkpoint---------')
            save_checkpoint(e, checkpoint_dir, net, opt)

        test_loss, test_acc = test(gen_test, loss, net, params['burnin_steps'], print_error = True)
        
        if not args.no_save:
            write_stats(e, test_acc, test_loss, writer)
            writer.add_scalar('/learning_rate', (opt.param_groups[-1]['lr']), e)

    train(gen_train, loss, net, opt, e, params['burnin_steps'], params['reg_l'])



