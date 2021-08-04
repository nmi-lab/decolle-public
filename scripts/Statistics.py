from decolle.utils import parse_args, prepare_experiment
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
import scipy.io as spio
import tqdm
from sklearn import svm, datasets, metrics

def main():
    np.set_printoptions(precision=4)
    # args = parse_args('parameters/params.yml')
    args = parse_args('parameters/params_MN.yml')
    # args = parse_args('parameters/params_MN_multipar.yml')
    # args = parse_args('parameters/params_MN_multipar2.yml')
    # args = parse_args('samples/params_to_test/0_a.yml')
    device = args.device

    path_to_save='C:/Users/stanzarella/OneDrive - Fondazione Istituto Italiano Tecnologia/Documents/IIT_Project/2.Codes/FromLiterature/decolle-public/scripts/stats/'

    starting_epoch = 0

    params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args=args)
    # log_dir = dirs['log_dir']
    # checkpoint_dir = dirs['checkpoint_dir']

    dataset = importlib.import_module(params['dataset'])
    try:
        create_data = dataset.create_data
    except AttributeError:
        create_data = dataset.create_dataloader

    verbose = args.verbose

    ## Load Data

    gen_train, gen_test = create_data(root=params['filename'],
                                      chunk_size_train=params['chunk_size_train'],
                                      chunk_size_test=params['chunk_size_test'],
                                      overlap_size_train_perc=params['overlap_size_train_perc'],
                                      overlap_size_test_perc=params['overlap_size_test_perc'],
                                      perc_test_norm=params['perc_test_norm'],
                                      muscle_to_exclude=params['muscle_to_exclude'],
                                      class_to_include=params['class_to_include'],
                                      thr_firing_excl_slice=params['thr_firing_excl_slice'],
                                      batch_size=params['batch_size'],
                                      dt=params['deltat'],
                                      num_workers=params['num_dl_workers'])

    # data_batch, target_batch = next(iter(gen_train))
    # print(data_batch)
    # print(target_batch)
    kernel = 'linear'  # 'rbf'
    Sv = []
    Dv = []
    Tv = []
    TrAccS = []
    TrAccD = []

    # S = np.zeros([data_batch.size()[3], data_batch.size()[0]])
    # D = np.zeros([data_batch.size()[3], data_batch.size()[0]])
    # T = np.zeros([data_batch.size()[3], data_batch.size()[0]])
    for e in range(starting_epoch, params['num_epochs']):

        # Train

        S_epo = []
        D_epo = []
        T_epo = []
        TrAccS_epo = []
        TrAccD_epo = []
        TrAccS_epo_test = []
        TrAccD_epo_test = []
        TrAccS_epo_test_cum = []
        TrAccD_epo_test_cum = []
        X_S_cum = []
        # X_D_cum = []
        Y_cum = []
        M = 0
        iter_gen_train = iter(gen_train)
        for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(e)):
            M = np.max([data_batch.size()[0], M])

            S = np.zeros([data_batch.size()[2], M])
            D = np.zeros([data_batch.size()[2], M])
            T = np.zeros([data_batch.size()[2], M])

            X_S = []
            X_D = []
            Y = []
            for jjj in range(0,data_batch.size()[0]): # Batches
                xs = []
                xd = []
                for iii in range(0, data_batch.size()[2]): # Neurons

                    S[iii,jjj] = int(data_batch[jjj,:,iii].sum())
                    xs.append(S[iii,jjj])

                    if S[iii,jjj] > 0 :
                        D[iii, jjj] = int( np.diff(np.where(data_batch[jjj, :, iii]==1)).sum()/S[iii,jjj] )
                        xd.append(D[iii, jjj])
                    else:
                        xd.append(0)

                    T[iii, jjj] = int(np.array(np.where(target_batch[jjj,iii,:]==1)))

                X_S.append(xs)
                X_D.append(xd)
                Y.append(T[0,jjj])
                X_S_cum.append(xs)
                # X_D_cum.append(xd)
                Y_cum.append(T[0, jjj])
            # SVM ---------------------------------------------------------------------------------------------------
            # Training only with this batch
            # Trained with spike counting
            clf_S = svm.SVC(kernel=kernel, C=1000)
            clf_S.fit(X_S, Y)
            y_pred_S = clf_S.predict(X_S)
            train_acc_S = metrics.accuracy_score(Y, y_pred_S)
            # Trained with ISI
            clf_D = svm.SVC(kernel=kernel, C=1000)
            clf_D.fit(X_D, Y)
            y_pred_D = clf_D.predict(X_D)
            train_acc_D = metrics.accuracy_score(Y, y_pred_D)

            # Training with this batch and the previous
            # Trained with spike counting
            clf_S_cum = svm.SVC(kernel=kernel, C=1000)
            clf_S_cum.fit(X_S_cum, Y_cum)
            # y_pred_S = clf_S.predict(X_S)
            # train_acc_S = metrics.accuracy_score(Y, y_pred_S)
            # Trained with ISI
            # clf_D_cum = svm.SVC(kernel='linear', C=1000)
            # clf_D_cum.fit(X_D_cum, Y_cum)
            # y_pred_D = clf_D.predict(X_D)
            # train_acc_D = metrics.accuracy_score(Y, y_pred_D)

            S_epo.append(S)
            D_epo.append(D)
            T_epo.append(T)
            TrAccS_epo.append(train_acc_S)
            TrAccD_epo.append(train_acc_D)

            ## Test per each train batch with all test batches
            train_acc_S_test = []
            train_acc_D_test = []
            train_acc_S_test_cum = []
            train_acc_D_test_cum = []
            iter_gen_test = iter(gen_test)
            for data_batch_test, target_batch_test in tqdm.tqdm(iter_gen_test, desc='Testing'):
                S_test = []
                D_test = []
                Y_test = []
                for jjj in range(0, data_batch_test.size()[0]):  # Batches
                    xs = []
                    xd = []
                    for iii in range(0, data_batch_test.size()[2]):  # Neurons
                        ss = int(data_batch_test[jjj,:,iii].sum())
                        xs.append(ss)
                        if ss > 0:
                            dd = int(np.diff(np.where(data_batch_test[jjj, :, iii] == 1)).sum() / ss)
                            xd.append(dd)
                        else:
                            xd.append(0)

                        tt = int(np.array(np.where(target_batch_test[jjj, iii, :] == 1)))
                    S_test.append(xs)
                    D_test.append(xd)
                    Y_test.append(tt)
                # Tested with spike counting
                y_pred_S_test = clf_S.predict(S_test)
                y_pred_S_test_cum = clf_S_cum.predict(S_test)
                if y_pred_S_test.__len__() == Y_test.__len__():
                    train_acc_S_test.append(metrics.accuracy_score(Y_test, y_pred_S_test))
                if y_pred_S_test_cum.__len__() == Y_test.__len__():
                    train_acc_S_test_cum.append(metrics.accuracy_score(Y_test, y_pred_S_test_cum))

                # Tested with ISI
                y_pred_D_test = clf_D.predict(D_test)
                # y_pred_D_test_cum = clf_D_cum.predict(D_test)
                if y_pred_D_test.__len__() == Y_test.__len__():
                    train_acc_D_test.append(metrics.accuracy_score(Y_test, y_pred_D_test))
                # if y_pred_D_test_cum.__len__() == Y_test.__len__():
                #     train_acc_D_test_cum.append(metrics.accuracy_score(Y_test, y_pred_D_test_cum))

            TrAccS_epo_test.append(train_acc_S_test)
            TrAccD_epo_test.append(train_acc_D_test)
            TrAccS_epo_test_cum.append(train_acc_S_test_cum)
            TrAccD_epo_test_cum.append(train_acc_D_test_cum)

        # SAVE
        struct = {'S': S_epo, 'D': D_epo, 'T': T_epo,'TrAccS':TrAccS_epo,'TrAccD':TrAccD_epo,'TrAccS_test':TrAccS_epo_test,'TrAccS_test_cum':TrAccS_epo_test_cum,'TrAccD_test':TrAccD_epo_test,'Params': params}
        spio.savemat(path_to_save + 'matlab_matrix_' + str(e) + '_'+ kernel.upper() +'.mat', struct)



    # struct = {'S':Sv,'D':Dv,'T':Tv,'TrAccS':TrAccS,'TrAccD':TrAccD,'Params':params}
    # spio.savemat(path_to_save + 'matlab_matrix.mat', struct)

    # data_batch = torch.Tensor(data_batch).to(device)
    # target_batch = torch.Tensor(target_batch).to(device)
if __name__ == '__main__':
    main()