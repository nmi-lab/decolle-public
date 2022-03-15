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
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib

from decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayer, LIFLayerRefractory, LIFLayerNonorm
from decolle.init_functions import init_LSUV_actrate
from decolle.utils import parse_args, train_timewrapped, test_timewrapped, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot, decolle_style_reg

np.set_printoptions(precision=4)

def main(params_file, name = None, save=False, extra_params={}):
    args = parse_args(params_file)
    device = args.device

    starting_epoch = 0

    if name is None:
        name = __file__.split('/')[-1].split('.')[0]
    params, writer, dirs = prepare_experiment(name=name, args = args)
    log_dir = dirs['log_dir']
    checkpoint_dir = dirs['checkpoint_dir']

    params.update(extra_params)

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
    data_batch = torch.tensor(data_batch).to(device)
    target_batch = torch.tensor(target_batch).to(device)

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
                        wrp=params['wrp'],
                        dropout=params['dropout'],
                        beta=params['beta'],
                        num_conv_layers=params['num_conv_layers'],
                        num_mlp_layers=params['num_mlp_layers'],
                        lc_ampl=params['lc_ampl'],
                        lif_layer_type = LIFLayer,
                        method=params['learning_method'],
                        with_output_layer=params['with_output_layer']).to(device)

    net.burnin = params['burnin_steps']

    loss = torch.nn.CrossEntropyLoss() #DECOLLELoss(net = net, loss_fn = loss, reg_l = params['reg_l'])

    if params['reg_l'] is not None:
        reg_loss_fn = decolle_style_reg(params['reg_l'])
    else:
        reg_loss_fn = lambda x: torch.Tensor(0.)

    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda epoch: 0.99 ** epoch)

    #loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
    #if net.with_output_layer:
    #    loss[-1] = cross_entropy_one_hot
    #decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)

    ##Initialize
    net.init_parameters(data_batch[:32])
    
    init_LSUV_actrate(net, data_batch[:32], params['act_rate'])

    test_acc_hist = []
    for e in range(starting_epoch , params['num_epochs'] ):
        total_loss, act_rate, train_acc = train_timewrapped(gen_train, loss, net, opt, e, reg_loss_fn = reg_loss_fn, batches_per_epoch=100)
        if save:
            writer.add_scalar('/train_acc/{0}'.format(0), train_acc[0], e)
            writer.add_scalar('/total_loss/{0}'.format(0), total_loss[0], e)
            for i in range(len(net)):
                writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
        else:
            print(act_rate)

        if (e%params['test_interval'])==0:
            test_loss, test_acc = test_timewrapped(gen_test, loss, net, print_error = True)
            test_acc_hist.append(test_acc)
    
            if save:
                write_stats(e, test_acc, test_loss, writer)
                np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)

        scheduler.step()
    
    del net
    return test_acc

if __name__ == "__main__":
    main(params_file = 'parameters/params_bptt_nmnist_mlp.yml')
