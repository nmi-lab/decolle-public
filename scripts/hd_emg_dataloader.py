#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Author: Simone Tanzarella
#
# Creation Date : Thu 04 Feb 2021 10:30:00 AM CEST
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
# -----------------------------------------------------------------------------

import struct
import time, copy
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from torchneuromorphic.neuromorphic_dataset import NeuromorphicDataset
from torchneuromorphic.events_timeslices import *
from torchneuromorphic.transforms import *
# from .create_hdf5 import create_events_hdf5
import scipy.io
from scipy import signal
import random


# mapping = {0: 'index_flx',
#            1: 'index_ext',
#            2: 'middle_flx',
#            3: 'middle_ext',
#            4: 'ring_flx',
#            5: 'little_flx',
#            6: 'little_ext',
#            7: 'thumb_flx',
#            8: 'thumb_ext'}

# muscles = ['FDI', 'IIDI', 'IIIDI', 'IVDI', 'ADM', 'FPB', 'APB', 'OPP', 'ECU', 'EDC', 'ECR', 'FCU', 'FDS', 'FCR']
fsamp = 2048 # The sample frequency of the original signal

# class ToCountFrameMN(object):
#     """Convert Address Events to Binary tensor.
#     Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T x C x H x W) in the range [0., 1., ...]
#     """
#     def __init__(self, T=500, size=[2, 32, 32]):
#         self.T = T
#         self.size = size
#
#     def __call__(self, tmad):
#
#         ts = range(0, self.T)
#         chunks = np.zeros([len(ts)] + self.size, dtype='int8')
#
#         times = tmad[:,0].astype(int)
#         if len(times) > 0 :
#             t_start = times[0]
#             t_end = times[-1]
#             if tmad.shape[1] == 2:
#                 addrs = tmad[:, 1].astype(int)
#             else:
#                 addrs = tmad[:,1:]
#
#             idx_start = 0
#             idx_end = 0
#             for i, t in enumerate(ts):
#                 idx_end += find_first(times[idx_end:], t+1)
#                 if idx_end > idx_start:
#                     ee = addrs[idx_start:idx_end]
#                     if len(ee.shape) == 1:
#                         i_pol_x_y = (i, ee)
#                     elif ee.shape[1] == 3:
#                         i_pol_x_y = (i, ee[:, 0], ee[:, 1], ee[:, 2])
#                     np.add.at(chunks, i_pol_x_y, 1)
#                 idx_start = idx_end
#         return chunks
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(T={0})'.format(self.T)

class MNDataset(NeuromorphicDataset):

    directory = 'data/hdsemg/'
    resources_local = [directory + 'Train', directory + 'Test']

    def __init__(
            self,
            root,
            train=True,
            download_and_create=True,
            chunk_size=500,
            overlap_perc=0,
            perc_test_norm=0.1,
            muscle_to_exclude=[],
            class_to_include=[],
            thr_firing_excl_slice=[],
            slices_to_take=[],
            dt=1000):


        self.n = 0
        # self.nclasses = self.num_classes = 9
        self.download_and_create = download_and_create
        self.root = root
        self.train = train
        self.dt = dt
        self.chunk_size = chunk_size
        self.ov=overlap_perc
        self.key_counter = 0
        self.batch_counter = 0
        self.muscle2Excl = muscle_to_exclude
        self.perc_test_norm = perc_test_norm
        self.class_to_include = class_to_include
        self.thr_firing_excl_slice = thr_firing_excl_slice
        self.equalize = False

        ## 1. LOAD THE FILE with the Keys
        matfile = scipy.io.loadmat(root)

        self.fsamp = 2048
        self.subj = root[(root.find('MNDS_')+5):(root.find('MNDS_')+9)]


        self.nCh = 64 # nChannels per grid

        # if len(self.muscle2Excl) == 0:
        self.nGrids = 6
        # else:
        #     self.nGrids = 6 - len(self.muscle2Excl)

        self.pinout = [[0, 12, 25, 38, 51],
                       [0, 13, 26, 39, 52],
                       [1, 14, 27, 40, 53],
                       [2, 15, 28, 41, 54],
                       [3, 16, 29, 42, 55],
                       [4, 17, 30, 43, 56],
                       [5, 18, 31, 44, 57],
                       [6, 19, 32, 45, 58],
                       [7, 20, 33, 46, 59],
                       [8, 21, 34, 47, 60],
                       [9, 22, 35, 48, 61],
                       [10, 23, 36, 49, 62],
                       [11, 24, 37, 50, 63]]
        self.chanRed = False


        ## 2. Extract here labels and keys

        EMG = matfile['emg'][:, 0]

        EMG, remChan = cleanData(EMG)

        labels = matfile['emg'][:, 1] - 1*(min(matfile['emg'][:, 1]) == 1)

        m = []
        for nc in range(0, int(np.max(labels))+1):
            # m.append(np.sum(labels == nc))
            m.append(np.sum( labels == nc , axis=1))
        if min(m) < max(m):
            self.equalize = True

        self.nclasses = int(np.max(labels)) + 1*(int(np.min(labels)) == 0)
        #Create new labels which go continuously from 0 to newNClasses

        # CLASS SELECTION HERE
        # keys = data ; l, labels = class labels
        if type(self.class_to_include) is list:
            if not(len(self.class_to_include) == 0):
                self.nclasses = 0
                l = []

                if self.chanRed:
                    # Chan reduction: one channel per column (5-fold reduction), since grids are 13x5
                    keys = np.zeros([0,len(self.pinout)*(self.nGrids - len(self.muscle2Excl))])
                else:
                    keys = np.zeros([0,self.nCh*(self.nGrids - len(self.muscle2Excl))])

                for nc in self.class_to_include:
                    # l.extend(labels[labels == nc])
                    # l.extend([int(self.nclasses) for iii in range(0, np.sum([labels == nc]))])
                    emgToAppend = EMG[np.argwhere(labels == nc)[:,1]]
                    for ccc in range(0,emgToAppend.shape[0]): # For each channel
                        if self.chanRed:
                            # Chan reduction: one channel per column (5-fold reduction), since grids are 13x5
                            redemg = np.zeros([emgToAppend[ccc].shape[1], keys.shape[1]])
                            gr2 = 0
                            for gr in range(0,self.nGrids):

                                # Grid (/channel) selection
                                if np.sum(np.array(self.muscle2Excl) == gr) == 0:
                                    for ch in range(0, len(self.pinout)):
                                        redemg[:,gr2*len(self.pinout)+ch] = np.mean(emgToAppend[ccc][gr*self.nCh+np.array(self.pinout[ch]),:],0)
                                    gr2 += 1
                            # Concatenation
                            keys = np.row_stack([keys, redemg])
                        else:
                            # Chan reduction: one channel per column (5-fold reduction), since grids are 13x5
                            hdemg = np.zeros([emgToAppend[ccc].shape[1], keys.shape[1]])
                            gr2 = 0
                            for gr in range(0, self.nGrids):

                                # Grid (/channel) selection
                                if np.sum(np.array(self.muscle2Excl) == gr) == 0:

                                    hdemg[:, gr2 * self.nCh + np.array(range(0,self.nCh))] = np.transpose(emgToAppend[ccc][gr * self.nCh + np.array(range(0,self.nCh)), :])
                                    gr2 += 1
                            # Concatenation
                            keys = np.row_stack([keys, hdemg])
                            # keys = np.row_stack([keys, np.transpose(emgToAppend[ccc])])
                        l.extend([int(self.nclasses) for iii in range(0, emgToAppend[ccc].shape[1])])
                    self.nclasses+=1
                labels = np.array(l)

                self.nGrids = self.nGrids - len(self.muscle2Excl)

        ## TODO: we need now to aggregate here, not to divide


        if self.train:
            ## Create TRAIN DATASET HERE

            ## TODO: AGGREGATING HERE
            self.aggregateWindows(keys,labels)

            # Prepare reshuffling train/test here!!!
            ## TODO: we need to reshuffle after aggregation
            self.n = self.labels.shape[0]

            if not (type(slices_to_take) is np.ndarray):
                slices_to_take = np.arange(0, self.n)
            elif slices_to_take.shape[0] == 0:
                slices_to_take = np.arange(0, self.n)
            # First reshuffle
            random.shuffle(slices_to_take)
            # Then divide train and test
            self.sliceTrain = slices_to_take[0:int(np.fix((1-self.perc_test_norm)*self.n))]
            self.sliceTest = slices_to_take[int(np.fix((1 - self.perc_test_norm) * self.n)):]

            # Reshuffling Train
            self.labelsTrain = self.labels[self.sliceTrain]
            self.n = self.labelsTrain.shape[0]
            self.keysTrain = self.tmp[self.sliceTrain]

            if self.equalize:
                self.keysTrain, self.labelsTrain = equalizeClasses(self.keysTrain, self.labelsTrain)
                self.n = self.labelsTrain.shape[0]

        else:
            ## Create TEST DATASET HERE
            #
            self.aggregateWindows(keys, labels)

            # Reshuffling Test
            self.labelsTest = self.labels[slices_to_take]
            self.keysTest = self.tmp[slices_to_take]
            self.n = self.labelsTest.shape[0]

            if self.equalize:
                self.keysTest, self.labelsTest = equalizeClasses(self.keysTest, self.labelsTest)
                self.n = self.labelsTest.shape[0]



        # ## 3. Create transform
        # transform, target_transform = self.createTransform(train)

        # super(MNDataset, self).__init__(
        #     root,
        #     transform=transform,
        #     target_transform=target_transform)
    #
    # def download(self):
    #     isexisting = super(MNDataset, self).download()

    def normEMG(self,tmp):
        # TODO: WE NEED TO GENERALIZE THIS NORMALIZATION!!!!
        for ngrids in range(0, self.nGrids):
            if self.chanRed:
                GR = tmp[:, ngrids * len(self.pinout):(ngrids + 1) * len(self.pinout)]
                tmp[:, ngrids * len(self.pinout):(ngrids + 1) * len(self.pinout)] = (GR - np.mean(GR)) / np.max(GR)
            else:
                GR = tmp[:, ngrids * self.nCh:(ngrids + 1) * self.nCh]
                tmp[:, ngrids * self.nCh:(ngrids + 1) * self.nCh] = (GR - np.min(GR)) / np.max(GR)
        return tmp

    def __len__(self):
        return self.n

    def aggregateWindows(self,keys,labels):

        tmp = np.zeros([0, keys.shape[1]])
        qqq = 0
        ppp = 0
        KK = np.zeros([1, keys.shape[1]])
        # chunk_size = window widht (buffer in myocontrol) in ms : example, chunk_size = 200 ms
        LL = np.zeros([int(np.fix(self.chunk_size / 50)), 1])
        newLab = np.zeros([0, 1])
        for indk, kkk in enumerate(keys):

            KK = KK + np.square(keys[qqq * int(np.fix(self.chunk_size / 50)) + ppp, :]) * np.fix(self.fsamp * 0.05)
            LL[ppp] = int(labels[qqq * int(np.fix(self.chunk_size / 50)) + ppp])
            if ppp < np.fix(self.chunk_size / 50) - 1:
                ppp += 1
            else:
                if np.sum(LL == LL[0]) == int(np.fix(self.chunk_size / 50)):
                    tmp = np.row_stack([tmp, np.sqrt(KK / (np.fix(self.chunk_size / 50) * np.fix(self.fsamp * 0.05)))])
                    # newLab = newLab.extend([int(LL/int(np.fix(self.chunk_size / 50))) for iii in range(0, 1)])
                    newLab = np.row_stack([newLab, int(labels[qqq * int(np.fix(self.chunk_size / 50)) + ppp])])
                # else:
                #     print(LL)
                #     print(qqq)
                #     print(ppp)
                ppp = 0
                qqq += 1
                KK = np.zeros([1, keys.shape[1]])
                LL = np.zeros([int(np.fix(self.chunk_size / 50)), 1])

        self.tmp = np.array(tmp)

        ## Data Normalization per each grid

        tmp = self.normEMG(tmp)

        # Same number of labels with respect to slices
        # labels = np.repeat(labels, int(self.nSlicesPerRep))
        self.labels = np.array(newLab)

    # def __getitem__(self, key):
    #
    #     if self.train:
    #         target = self.labelsTrain[key]
    #         key = self.keysTrain[key]
    #     else:
    #         target = self.labelsTest[key]
    #         key = self.keysTest[key]
    #
    #     data = self.sample(key, T=int(np.fix(self.chunk_size/self.dt*self.fsamp)), ov=int(np.fix(self.chunk_size*self.ov/self.dt*self.fsamp)))
    #
    #     # if self.batch_counter > 0 & np.mod(self.batch_counter,)
    #
    #     if self.transform is not None:
    #         data = self.transform(data)
    #
    #
    #     if self.target_transform is not None:
    #         target = self.target_transform(target)
    #
    #
    #     return data, target

    # def sample(self, key, T=300, ov=0):
    #
    #     times = []
    #     addr = []
    #     mmm = 0 # Counter of motor neuron number -> Address, start from 0
    #     NUMFIRINGS = 0 # Count number total firings in the chunk
    #
    #     # Run across muscles
    #     for iii in range(0, len(self.muscles)):
    #         # Exclude muscles in self.muscle2Excl
    #         if (np.argwhere(np.array(self.muscle2Excl) == iii + 1)).shape[0] == 0:
    #             # If contains motor neurons
    #             if self.data[iii, 0].shape[1] > 0:
    #                 # Run across motorneurons in the muscle
    #                 for jjj in range(0, self.data[iii, 0].shape[0]):
    #                     if (key[0]+self.key_counter*T+T) <= key[1]:
    #                         # Equivalent of "get_tmad_slice", but we can try also with that function
    #                         times.extend(self.data[iii, 0][jjj, 0][(self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T))])
    #                         addr.extend([mmm for _ in range(0, np.sum((self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T))))])
    #                         NUMFIRINGS += np.sum((self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T)))
    #                         mmm += 1
    #
    #     # Remove here times, addr items less than firing thr
    #     if NUMFIRINGS >= np.fix(self.chunk_size / 1000 * self.thr_firing_excl_slice * self.nMN / 10):
    #         ADDR = np.array(addr)
    #         addr = ADDR[np.argsort(times)] #.tolist()
    #         times = np.sort(times) #.tolist()
    #         if len(times) > 0 :
    #             times -= times[0]
    #
    #     tmad = np.column_stack([times, addr]).astype(int)
    #     # self.key_counter += 1
    #
    #     return tmad

    # def createTransform(self, train):
    #
    #     downsampfact = self.fsamp / self.dt
    #
    #     if train:
    #
    #         transform = Compose([
    #             Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
    #             ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
    #             ToTensor()])
    #
    #         target_transform = Compose([Repeat(self.chunk_size), toOneHot(self.nclasses)])
    #     else:
    #
    #         transform = Compose([
    #             Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
    #             ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
    #             ToTensor()])
    #
    #         target_transform = Compose([Repeat(self.chunk_size), toOneHot(self.nclasses)])
    #
    #     return transform, target_transform

    # def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
    #     """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.
    #     Args:
    #         X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
    #         y: The labels
    #     """
    #
    #     labels_ = np.array(y, dtype=np.float)
    #     number_of_batches = len(X) // batch_size
    #     sample_index = np.arange(len(X))
    #
    #     # compute discrete firing times
    #     tau_eff = 20e-3 / time_step
    #     firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.float)
    #     unit_numbers = np.arange(nb_units[1] * nb_units[2])
    #     # unit_numbers_2 = np.arange(nb_units[2])
    #
    #     if shuffle:
    #         np.random.shuffle(sample_index)
    #
    #     # total_batch_count = 0
    #     counter = 0
    #     while counter < number_of_batches:
    #         batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
    #
    #         coo = [[] for i in range(5)]
    #         for bc, idx in enumerate(batch_index):
    #             for n_mat in range(0, nb_units[0]):
    #                 c = firing_times[idx] < nb_steps
    #                 times, units = firing_times[idx][c], unit_numbers[c]
    #
    #                 cols = np.fix((units + 1) / nb_units[1] - 1).astype(int)  # cols
    #                 rows = (np.mod((units), nb_units[1])).astype(int)  # rows
    #
    #                 batch = [bc for _ in range(len(times))]
    #                 matrices = [n_mat for _ in range(len(times))]
    #                 coo[0].extend(batch)
    #                 coo[1].extend(times)
    #                 coo[2].extend(matrices)
    #                 coo[3].extend(rows)
    #                 coo[4].extend(cols)
    #
    #         i = torch.LongTensor(coo).to(device)
    #         v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    #
    #         X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
    #             [batch_size, nb_steps, nb_units[0], nb_units[1], nb_units[2]])).to(device)
    #         # y_batch = torch.tensor(labels_[batch_index],device=device)
    #         LL = []
    #         for lab in labels_[batch_index]:
    #             LL.append([[lab]])
    #         y_batch = torch.tensor(np.repeat(np.array(LL), 300, axis=1), device=device)
    #
    #         yield X_batch.detach().cpu().to_dense().numpy(), y_batch.detach().cpu().numpy()
    #
    #         counter += 1
def equalizeClasses(keys,labels):
    m = np.inf
    for nc in range(0, int(max(labels))+1):
        m = np.min([np.sum(labels == nc), m])
    m = int(m)
    kk = np.zeros([0,  keys.shape[1]])
    ll = []
    for nc in range(0, int(max(labels))+1):
        kk = np.row_stack([kk,  keys[np.argwhere(labels == nc)[:,0]][0:m] ])
        ll.extend([nc for iii in range(0, m)])
    slices_to_take = np.arange(0, len(ll))
    random.shuffle(slices_to_take)
    ll = np.array(ll)
    # kk = kk.astype(int)
    ll = ll[slices_to_take]
    kk = kk[slices_to_take]
    return kk, ll

def cleanData(EMG):
    SIG = np.zeros([0,70])
    remChan = []
    for iii in range(0,EMG.shape[0]):
        if EMG[iii].shape[1] >= 70:
            SIG = np.row_stack([SIG,EMG[iii][:, 0:70]])

    for iii in range(0,EMG.shape[0]):

        IND1 = RMS(EMG[iii], 1) > 10 * np.mean(RMS(SIG[:, 24: 70], 1))
        IND2 = np.zeros([EMG[iii].shape[0],1]).astype('bool')
        for jjj in range(0,EMG[iii].shape[0]):
            try:
                IND2[jjj] = np.sum(EMG[iii][jjj,:] > 7.5 * np.mean(EMG[iii][jjj,:])) > 0
            except:
                print(EMG[iii].shape)
        IND2 = np.squeeze(IND2,1)
        EMG[iii][IND1 | IND2,:] = np.zeros([np.sum(IND1 | IND2), EMG[iii].shape[1]])
        remChan.append(np.argwhere(IND1 | IND2))

    remChan = np.array(remChan)

    return EMG, remChan

def RMS(sig,dim):

    X = np.sqrt(np.mean(np.square(sig),dim),dim)

    return X

def create_datasets(
        root='data/motoneurons/MNDS_KaJu.mat',
        batch_size=72,
        chunk_size_train=300,
        chunk_size_test=300,
        overlap_size_train_perc = 0,
        overlap_size_test_perc = 0,
        perc_test_norm=0.1,
        muscle_to_exclude=[],
        class_to_include=[],
        thr_firing_excl_slice=[],
        ds=1,
        dt=1000):

    train_ds = MNDataset(root, train=True,
                             chunk_size=chunk_size_train,
                             overlap_perc=overlap_size_train_perc,
                             perc_test_norm=perc_test_norm,
                             muscle_to_exclude=muscle_to_exclude,
                             class_to_include=class_to_include,
                             thr_firing_excl_slice=thr_firing_excl_slice,
                             slices_to_take=[],
                             dt=dt)

    test_ds = MNDataset(root, train=False,
                            chunk_size=chunk_size_train,
                            overlap_perc=overlap_size_train_perc,
                            perc_test_norm=perc_test_norm,
                            muscle_to_exclude=muscle_to_exclude,
                            class_to_include=class_to_include,
                            thr_firing_excl_slice=thr_firing_excl_slice,
                            slices_to_take=train_ds.sliceTest,
                            dt=dt)

# train_ds = MNDataset(root, train=True,
#                          transform=transform_train,
#                          target_transform=target_transform_train,
#                          chunk_size=chunk_size_train,
#                          dt=dt)
#
# test_ds = MNDataset(root, transform=transform_test,
#                         target_transform=target_transform_test,
#                         train=False,
#                         chunk_size=chunk_size_test,
#                         dt=dt)

    return train_ds, test_ds


def create_dataloader(
        root='data/motoneurons/MNDS_KaJu.mat',
        batch_size=72,
        chunk_size_train=300,
        chunk_size_test=300,
        overlap_size_train_perc = 0,
        overlap_size_test_perc = 0,
        perc_test_norm=0.1,
        muscle_to_exclude=[],
        class_to_include=[],
        thr_firing_excl_slice=[],
        ds=1,
        dt=1000,
        **dl_kwargs):
    train_d, test_d = create_datasets(
        root=root,
        batch_size=batch_size,
        chunk_size_train=chunk_size_train,
        chunk_size_test=chunk_size_train,
        overlap_size_train_perc=overlap_size_train_perc,
        overlap_size_test_perc=overlap_size_train_perc,
        perc_test_norm=perc_test_norm,
        muscle_to_exclude=muscle_to_exclude,
        class_to_include=class_to_include,
        thr_firing_excl_slice=thr_firing_excl_slice,
        ds=ds,
        dt=dt)

    # TODO: Decomment and debug
    # # # Commented (Ctrl+1) by Simone Tanzarella 19/01/2021
    # train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    # # # Modified by Simone Tanzarella 19/01/2021
    # # # train_dl = torch.utils.data.DataLoader(train_d, shuffle=False, batch_size=batch_size, **dl_kwargs)
    # test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    # return train_dl, test_dl
    return train_d, test_d