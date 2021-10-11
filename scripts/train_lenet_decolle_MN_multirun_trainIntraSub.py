#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_lenet_decolle
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified : Simone Tanzarella, Sept 28. 2021
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
from decolle.lenet_decolle_model import LenetDECOLLE, DECOLLELoss, LIFLayerVariableTau, LIFLayer
from decolle.lenet_decolle_model_1D_MN import LenetDECOLLE1DMN, DECOLLELoss, LIFLayerVariableTau, LIFLayer
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib


def main():
    # fold = 'C:/Users/stanzarella/data/planned_params/FirstDefinitiveRun/'
    # file = 'params_MN_5cl_allcycle_cs150_ov0.5_extr.yml'
    np.set_printoptions(precision=4)
    args = parse_args('parameters/params_MN_10cl_incr_cs200_ov0_KaJu.yml')
    # path_to_save_sl4tr = os.path.curdir + '/slices_for_training/'

    # We do not have NVIDIA cuda in this laptop, uncomment the following line and comment the second when we have it
    # device = args.device
    device = 'cpu'

    # Do you want a final output layer?
    with_output_layer = False

    CV = 1 # For cross validation

    for cv in range(0, CV):
        ## Extract parameters and set writer
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

        #TODO: 1. Here we should add and insert a parameter (in params) to take only a portion of the data,
        # and then dividing this selected subset in train and test datasets; insert also info on
        # previous selected slices; add also a parameter to inform if it's hyper-parameters training or
        # dataset selection optimization
        if 'slices_for_train' not in params.keys():
            params['slices_for_train'] = []
        elif type(params['slices_for_train']) is not dict:
            params['slices_for_train'] = []
        else:
            if 'folder' not in params['slices_for_train'].keys():
                params['slices_for_train']['folder'] = 'slices_for_training'
            if 'folder' not in params['slices_for_train'].keys():
                params['slices_for_train']['type_train'] = 'hyper_parameters'
            if 'folder' not in params['slices_for_train'].keys():
                params['slices_for_train']['perc_hyperpar'] = 0.4

        gen_train, gen_test = create_data(root=params['filename'],
                                          chunk_size_train=params['chunk_size_train'],
                                          chunk_size_test=params['chunk_size_test'],
                                          overlap_size_train_perc=params['overlap_size_train_perc'],
                                          overlap_size_test_perc=params['overlap_size_test_perc'],
                                          perc_test_norm=params['perc_test_norm'],
                                          slices_for_train_in=params['slices_for_train'],
                                          muscle_to_exclude=params['muscle_to_exclude'],
                                          class_to_include=params['class_to_include'],
                                          thr_firing_excl_slice=params['thr_firing_excl_slice'],
                                          batch_size=params['batch_size'],
                                          dt=params['deltat'],
                                          num_workers=params['num_dl_workers'])

        #TODO: 2. We should keep trace of everything in the dataloader, in particular the sliceTrain and sliceTest
        # should be referred to the original entire dataset. Save these slices here, if it's the first time we selected them
        

        ## Create batches
        data_batch, target_batch = next(iter(gen_train))
        data_batch = torch.Tensor(data_batch).to(device)
        target_batch = torch.Tensor(target_batch).to(device)

        #d, t = next(iter(gen_train))
        input_shape = data_batch.shape[-3:]

        #Backward compatibility
        if 'dropout' not in params.keys():
            params['dropout'] = [.5]

        if len(params['input_shape']) == 0:
            params['input_shape'] = [0]

        if params['input_shape'][0] == 0:
            params['input_shape'] = data_batch.shape[-len(params['input_shape']):]

        if type(params['alpha']) is not list:
            params['alpha'] = [params['alpha']]
        if type(params['beta']) is not list:
            params['beta'] = [params['beta']]


        ## Create Model, Optimizer and Loss
        net = LenetDECOLLE1DMN(out_channels=params['out_channels'],
                           Nhid=np.array(params['Nhid']),
                           Mhid=np.array(params['Mhid']),
                           kernel_size=params['kernel_size'],
                           pool_size=params['pool_size'],
                           input_shape=params['input_shape'],
                           alpha=np.array(params['alpha']),
                           alpharp=np.array(params['alpharp']),
                           dropout=params['dropout'],
                           beta=np.array(params['beta']),
                           num_conv_layers=params['num_conv_layers'],
                           num_mlp_layers=params['num_mlp_layers'],
                           lc_ampl=params['lc_ampl'],
                           lif_layer_type=LIFLayer,
                           method=params['learning_method'],
                           with_output_layer=with_output_layer).to(device)
        # Check parameters
        net, opt, decolle_loss = check_parameters(params,net,args,data_batch,checkpoint_dir)


        # Start the training
        print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))
        # --------TRAINING LOOP----------
        if not args.no_train:
            test_acc_hist = []
            for e in range(starting_epoch , params['num_epochs'] ):
                interval = e // params['lr_drop_interval']

                opt = opt_manager(opt,interval,params)

                # TEST data every test_interval epochs
                if (e % params['test_interval']) == 0 and e!=0:
                    print('---------------Epoch {}-------------'.format(e))
                    if not args.no_save:
                        print('---------Saving checkpoint---------')
                        save_checkpoint(e, checkpoint_dir, net, opt)
                    ## TEST
                    test_loss, test_acc = test(gen_test, decolle_loss, net, params['burnin_steps'], print_error = True)
                    test_acc_hist.append(test_acc)
                    # Save stats
                    if not args.no_save:
                        write_stats(e, test_acc, test_loss, writer)
                        np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)

                ## TRAIN
                total_loss, act_rate = train(gen_train, decolle_loss, net, opt, e, params['burnin_steps'], online_update=params['online_update'])
                # Save stats
                if not args.no_save:
                    for i in range(len(net)):
                        writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
                        writer.add_scalar('/total_loss/{0}'.format(i), total_loss[i], e)
            starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)


## Functions

def check_parameters(params,net,args,data_batch,checkpoint_dir):
    if hasattr(params['learning_rate'], '__len__'):
        from decolle.utils import MultiOpt
        opts = []
        for i in range(len(params['learning_rate'])):
            opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i],
                                           betas=params['betas']))
        opt = MultiOpt(*opts)
    else:
        opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])

    reg_l = params['reg_l'] if 'reg_l' in params else None

    if 'loss_scope' in params and params['loss_scope'] == 'crbp':
        from decolle.lenet_decolle_model import CRBPLoss
        loss = torch.nn.SmoothL1Loss(reduction='none')
        decolle_loss = CRBPLoss(net=net, loss_fn=loss, reg_l=reg_l)
    else:
        loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
        if net.with_output_layer:
            loss[-1] = cross_entropy_one_hot
        decolle_loss = DECOLLELoss(net=net, loss_fn=loss, reg_l=reg_l)

    ##Initialize
    net.init_parameters(data_batch)

    ##Resume if necessary
    if args.resume_from is not None:
        print("Checkpoint directory " + checkpoint_dir)
        print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))
        if not os.path.exists(checkpoint_dir) and not args.no_save:
            os.makedirs(checkpoint_dir)

    # Printing parameters
    if args.verbose:
        print('Using the following parameters:')
        m = max(len(x) for x in params)
        for k, v in zip(params.keys(), params.values()):
            print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

    return net, opt, decolle_loss

def opt_manager(opt,interval,params):
    lr = opt.param_groups[-1]['lr']
    if interval > 0:
        print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
        opt.param_groups[-1]['lr'] = np.array(params['learning_rate']) / (interval * params['lr_drop_factor'])
    else:
        print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
        opt.param_groups[-1]['lr'] = np.array(params['learning_rate'])
    return opt

if __name__ == '__main__':
    main()