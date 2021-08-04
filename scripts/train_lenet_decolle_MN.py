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
from decolle.lenet_decolle_model_1D_MN import LenetDECOLLE1DMN, DECOLLELoss, LIFLayerVariableTau, LIFLayer
from decolle.utils import parse_args, train, test, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib

def main():
    np.set_printoptions(precision=4)
    # args = parse_args('parameters/params.yml')
    args = parse_args('parameters/params_MN.yml')
    # args = parse_args('parameters/params_MN_multipar.yml')
    # args = parse_args('parameters/params_MN_multipar2.yml')
    device = args.device


    starting_epoch = 0

    params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
    log_dir = dirs['log_dir']
    checkpoint_dir = dirs['checkpoint_dir']

    # Here the for loop to change parameters
    if 'Nhid_step' in params.keys():
        Nhid_step = params['Nhid_step']
    else:
        Nhid_step = [0]
    if 'Mhid_step' in params.keys():
        Mhid_step = params['Mhid_step']
    else:
        Mhid_step = [0]
    if 'alpha_step' in params.keys():
        alpha_step = params['alpha_step']
    else:
        alpha_step = [0]
    if 'beta_step' in params.keys():
        beta_step = params['beta_step']
    else:
        beta_step = [0]
    if 'overlap_size_train_perc' in params.keys():
        overlap = params['overlap_size_train_perc']
    else:
        overlap = [0]

    for ns in Nhid_step:
        for ms in Mhid_step:
            for ast in alpha_step:
                for bst in beta_step:
                    for T_tr in params['chunk_size_train']:
                        for T_te in params['chunk_size_test']:
                            for ov in overlap:

                                dataset = importlib.import_module(params['dataset'])
                                try:
                                    create_data = dataset.create_data
                                except AttributeError:
                                    create_data = dataset.create_dataloader

                                verbose = args.verbose

                                ## Load Data
                                # if 'overlap_size_train_perc' in params.keys():
                                gen_train, gen_test = create_data(chunk_size_train=T_tr,
                                                                  chunk_size_test=T_te,
                                                                  overlap_size_train_perc=ov,
                                                                  overlap_size_test_perc=ov,
                                                                  muscle_to_exclude=params['muscle_to_exclude'],
                                                                  batch_size=params['batch_size'],
                                                                  dt=params['deltat'],
                                                                  num_workers=params['num_dl_workers'])

                                # else:
                                #     gen_train, gen_test = create_data(chunk_size_train=T_tr,
                                #                                       chunk_size_test=T_te,
                                #                                       batch_size=params['batch_size'],
                                #                                       dt=params['deltat'],
                                #                                       num_workers=params['num_dl_workers'])

                                data_batch, target_batch = next(iter(gen_train))
                                data_batch = torch.Tensor(data_batch).to(device)
                                target_batch = torch.Tensor(target_batch).to(device)

                                #d, t = next(iter(gen_train))
                                input_shape = data_batch.shape[-3:]

                                #Backward compatibility
                                if 'dropout' not in params.keys():
                                    params['dropout'] = [.5]

                                if params['input_shape'][0] == 0:
                                    params['input_shape'] = data_batch.shape[-len(params['input_shape']):]

                                ## Create Model, Optimizer and Loss
                                net = LenetDECOLLE1DMN(out_channels=params['out_channels'],
                                                   Nhid=np.array(params['Nhid'])+ns,
                                                   Mhid=np.array(params['Mhid'])+ms,
                                                   kernel_size=params['kernel_size'],
                                                   pool_size=params['pool_size'],
                                                   input_shape=params['input_shape'],
                                                   alpha=np.array(params['alpha'])+ast,
                                                   alpharp=np.array(params['alpharp'])+ast,
                                                   dropout=params['dropout'],
                                                   beta=np.array(params['beta'])+bst,
                                                   num_conv_layers=params['num_conv_layers'],
                                                   num_mlp_layers=params['num_mlp_layers'],
                                                   lc_ampl=params['lc_ampl'],
                                                   lif_layer_type=LIFLayer,
                                                   method=params['learning_method'],
                                                   with_output_layer=True).to(device)
                                # else:
                                #     ## Create Model, Optimizer and Loss
                                #     net = LenetDECOLLE( out_channels=params['out_channels'],
                                #                         Nhid=params['Nhid'],
                                #                         Mhid=params['Mhid'],
                                #                         kernel_size=params['kernel_size'],
                                #                         pool_size=params['pool_size'],
                                #                         input_shape=params['input_shape'],
                                #                         alpha=params['alpha'],
                                #                         alpharp=params['alpharp'],
                                #                         dropout=params['dropout'],
                                #                         beta=params['beta'],
                                #                         num_conv_layers=params['num_conv_layers'],
                                #                         num_mlp_layers=params['num_mlp_layers'],
                                #                         lc_ampl=params['lc_ampl'],
                                #                         lif_layer_type = LIFLayer,
                                #                         method=params['learning_method'],
                                #                         with_output_layer=True).to(device)

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

                                print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

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
                                                writer.add_scalar('/total_loss/{0}'.format(i), total_loss[i], e)
                                    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)

if __name__ == '__main__':
    main()