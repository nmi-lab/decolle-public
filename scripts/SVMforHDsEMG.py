from decolle.utils import parse_args, prepare_experiment
import datetime, os, socket, tqdm
import numpy as np
import torch
import importlib
import scipy.io as spio
import tqdm
from sklearn import svm, datasets, metrics

def main():

    main_path = 'C:/Users/tanza/Documents/IIT-PostDoc/2.Codes/FromLiterature'
    script_path = '/decolle-public/scripts'
    # main_path = ''
    # script_path = ''

    FILES = os.listdir(main_path + script_path + '/params_to_test')

    for file in FILES:
        file = '000005.yml'
        np.set_printoptions(precision=4)
        args = parse_args('params_to_test/' + file)
        # args = parse_args('parameters/params.yml')
        # args = parse_args('parameters/params_MN.yml')
        # args = parse_args('parameters/params_MN_multipar.yml')
        # args = parse_args('parameters/params_MN_multipar2.yml')
        # args = parse_args('samples/params_to_test/0_a.yml')
        device = args.device

        path_to_save= main_path + script_path + '/SVMallSubjEMGnew/'

        starting_epoch = 0

        # SVM Regularization parameter
        C = 10000

        params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args=args)
        # log_dir = dirs['log_dir']
        # checkpoint_dir = dirs['checkpoint_dir']
        params['batch_size'] = 10

        dataset = importlib.import_module(params['dataset'])
        try:
            create_data = dataset.create_data
        except AttributeError:
            create_data = dataset.create_dataloader

        verbose = args.verbose

        for e in range(0, 5):  # range(starting_epoch, params['num_epochs']):
            ## Load Data
            print('CV ' + str(e))
            print('Load')

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

            NbatchTrain = int(np.shape(gen_train.keysTrain)[0]/params['batch_size'])
            NbatchTest = int(np.shape(gen_test.keysTest)[0] / params['batch_size'])

            # SVM Kernel
            kernel = 'linear'  # 'rbf' # 'linear'  #

            # Train

            # TrAccS_epo = []

            TrAccS_epo_cum = []

            # TrAccS_epo_test = []

            TrAccS_epo_test_cum = []


            # # for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(e)):
            # for nbtr in range(0,NbatchTrain):

            #data_batch = gen_train.keysTrain #[nbtr*params['batch_size']:(nbtr+1)*params['batch_size'],:]
            #target_batch = gen_train.labelsTrain #[nbtr*params['batch_size']:(nbtr+1)*params['batch_size']]

            data_batch_cum = gen_train.keysTrain #[0:(nbtr+1)*params['batch_size']]
            target_batch_cum = gen_train.labelsTrain #[0:(nbtr+1)*params['batch_size']]

            # SVM ---------------------------------------------------------------------------------------------------
            # # Training only with this batch
            # # Trained with spike counting
            # clf_S = svm.SVC(kernel=kernel, C=C) # C was 1000 for MNs
            # clf_S.fit(data_batch, target_batch)
            # y_pred_S = clf_S.predict(data_batch)
            # train_acc_S = metrics.accuracy_score(np.ravel(target_batch), np.ravel(y_pred_S))

            # Training with this batch and the previous
            # Trained with spike counting
            clf_S_cum = svm.SVC(kernel=kernel, C=C)
            clf_S_cum.fit(data_batch_cum, target_batch_cum)
            y_pred_S_cum = clf_S_cum.predict(data_batch_cum)
            train_acc_S_cum = metrics.accuracy_score(np.ravel(target_batch_cum), np.ravel(y_pred_S_cum))

            # TrAccS_epo.append(train_acc_S)
            TrAccS_epo_cum.append(train_acc_S_cum)

            ## Test per each train batch with all test batches
            train_acc_S_test = []
            train_acc_S_test_cum = []

            # # for data_batch_test, target_batch_test in tqdm.tqdm(iter_gen_test, desc='Testing'):
            # for nbte in range(0, NbatchTest):

            data_batch_test = gen_test.keysTest #[nbte * params['batch_size']:(nbte + 1) * params['batch_size'], :]
            target_batch_test = gen_test.labelsTest #[nbte * params['batch_size']:(nbte + 1) * params['batch_size']]

            # data_batch_test_cum = gen_test.keysTest[0:(nbte + 1) * params['batch_size'], :]
            # target_batch_test_cum = gen_test.labelsTest[0:(nbte + 1) * params['batch_size'], :]


            # Tested with spike counting
            # y_pred_S_test = clf_S.predict(data_batch_test)
            y_pred_S_test_cum = clf_S_cum.predict(data_batch_test)
            # if y_pred_S_test.__len__() == target_batch_test.__len__():
            #     ACC = metrics.accuracy_score(np.ravel(target_batch_test), np.ravel(y_pred_S_test))
            #     train_acc_S_test.append(ACC)
            # else:
            #     print('Predicted Y, less than targets')
            if y_pred_S_test_cum.__len__() == target_batch_test.__len__():
                ACC = metrics.accuracy_score(np.ravel(target_batch_test), np.ravel(y_pred_S_test_cum))
                train_acc_S_test_cum.append(ACC)
            else:
                print('Predicted Y, less than targets')


            # TrAccS_epo_test.append(train_acc_S_test)
            TrAccS_epo_test_cum.append(train_acc_S_test_cum)

            # SAVE
            struct = {'TrainSet': gen_train.keysTrain,'LabelsTrain': gen_train.labelsTrain,'TestSet': gen_test.keysTest,'LabelsTest':gen_test.labelsTest,'Ypred_test': y_pred_S_test_cum, 'TrAccS_test_cum': TrAccS_epo_test_cum,
                      'Params': params,'TrAccS_cum':TrAccS_epo_cum } #,
                      # 'TrAccS': TrAccS_epo, 'TrAccS_test': TrAccS_epo_test}
            spio.savemat(path_to_save + 'matlab_matrix_' + str(e) + '_' + kernel.upper() + '_' + file[0:-4] + '.mat',
                         struct)

if __name__ == '__main__':
    main()