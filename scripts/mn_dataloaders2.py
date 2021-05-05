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

class ToCountFrameMN(object):
    """Convert Address Events to Binary tensor.
    Converts a numpy.ndarray (T x H x W x C) to a torch.FloatTensor of shape (T x C x H x W) in the range [0., 1., ...]
    """
    def __init__(self, T=500, size=[2, 32, 32]):
        self.T = T
        self.size = size

    def __call__(self, tmad):

        ts = range(0, self.T)
        chunks = np.zeros([len(ts)] + self.size, dtype='int8')

        times = tmad[:,0].astype(int)
        if len(times) > 0 :
            t_start = times[0]
            t_end = times[-1]
            if tmad.shape[1] == 2:
                addrs = tmad[:, 1].astype(int)
            else:
                addrs = tmad[:,1:]

            idx_start = 0
            idx_end = 0
            for i, t in enumerate(ts):
                idx_end += find_first(times[idx_end:], t+1)
                if idx_end > idx_start:
                    ee = addrs[idx_start:idx_end]
                    if len(ee.shape) == 1:
                        i_pol_x_y = (i, ee)
                    elif ee.shape[1] == 3:
                        i_pol_x_y = (i, ee[:, 0], ee[:, 1], ee[:, 2])
                    np.add.at(chunks, i_pol_x_y, 1)
                idx_start = idx_end
        return chunks

    def __repr__(self):
        return self.__class__.__name__ + '(T={0})'.format(self.T)

class MNDataset(NeuromorphicDataset):

    directory = 'data/motoneurons/'
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

        # TODO: 1. randomised assignation of test and train portions, after slicing dataset (indicate from outside percentage Tr/Te)
        # DONE
        # TODO: 2. option for class exclusion
        # DONE, to improve
        # TODO: 3. threshold of firings to exclude a slice (imposed by outside)
        # DONE
        # TODO: 4.  substitute len(mapping) and self.nclasses, self.num_classes
        # DONE

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

        ## 1. LOAD THE FILE
        matfile = scipy.io.loadmat(root)
        self.data = matfile['e']['mu'][0, 0][0, 0][1]
        self.muscles = []
        for iii in range(0, matfile['e']['muscles'][0, 0].shape[0]):
            # TODO: : Insert here the condition for muscles to exclude (DONE)
            if (np.argwhere(np.array(self.muscle2Excl) == iii + 1)).shape[0] == 0:
                self.muscles.append(matfile['e']['muscles'][0, 0][iii, 0][0])

        self.fsamp = int(matfile['e']['fsamp'])

        print('Number of considered muscles : ' + str(self.data.shape[0]))
        print('Number of motor neurons per muscle :')
        self.nMN = 0
        for iii in range(0, self.data.shape[0]):
            # TODO: : Insert here the condition for muscles to exclude (DONE)
            if (np.argwhere(np.array(self.muscle2Excl)==iii+1)).shape[0] == 0:
                if self.data[iii, 0].shape[1] > 0:
                    self.nMN += self.data[iii, 0].shape[0]
                    print(self.muscles[iii] + ', ' + str(self.data[iii, 0].shape[0]) + ' MNs')
                else:
                    print(self.muscles[iii] + ', ' + str(self.data[iii, 0].shape[1]) + ' MNs')

        ## 2. Extract here labels and keys

        # TODO : dynamic assignation test and train portions (DONE)
        # self.labelsTrain = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] == 0, 2]-1
        # self.labelsTest = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] > 0, 2]-1
        # self.keysTrain = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] == 0, 0:2]
        # self.keysTest = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] > 0, 0:2]
        KEYS = matfile['e']['Keys'][0, 0]

        labels = KEYS[:, 2] - 1*(min(KEYS[:, 2]) == 1)
        keys = KEYS[:, 0:2]

        m = []
        for nc in range(0, max(labels)+1):
            m.append(np.sum(labels == nc))
        if min(m) < max(m):
            self.equalize = True

        # ## Exclude classes at this level
        # if self.class_to_include is list:
        #     for ccc in range(0,len(self.class_to_include)):
        #         KEYS[KEYS[:, 2]==self.class_to_include[ccc]]=[]
        # Take only the classes you need

        self.nclasses = max(labels) + 1*(min(labels) == 0)
        #Create new labels which go continuously from 0 to newNClasses
        keys = np.array(keys)
        # CLASS SELECTION HERE
        if type(self.class_to_include) is list:
            if not(len(self.class_to_include) == 0):
                self.nclasses = 0
                l = []
                k = np.zeros([0,keys.shape[1]])
                for nc in self.class_to_include:
                    # l.extend(labels[labels == nc])
                    l.extend([int(self.nclasses) for iii in range(0, np.sum([labels == nc]))])
                    k = np.row_stack([k, keys[labels == nc,: ]])
                    self.nclasses+=1
                labels = np.array(l)
                keys = k.astype(int)
        # Set number of slices to cut
        self.repWidth = int(np.fix(np.mean(np.diff(keys, 1, 1))))
        self.SF = self.dt /self.fsamp
        self.nSlicesPerRep = int(np.fix((self.repWidth-(self.chunk_size*self.ov)/self.SF) / ((self.chunk_size*(1-self.ov))/self.SF)))

        self.n = int(self.nSlicesPerRep * labels.shape[0])
        labels = np.repeat(labels, int(self.nSlicesPerRep))

        if self.train:
            ## Create TRAIN DATASET HERE

            # Prepare reshuffling train/test here!!!
            if not (type(slices_to_take) is np.ndarray):
                slices_to_take = np.arange(0, self.n)
            elif slices_to_take.shape[0] == 0:
                slices_to_take = np.arange(0, self.n)
            random.shuffle(slices_to_take)
            self.sliceTrain = slices_to_take[0:int(np.fix((1-self.perc_test_norm)*self.n))]
            self.sliceTest = slices_to_take[int(np.fix((1 - self.perc_test_norm) * self.n)):]

            # SLICING HERE
            tmp = []
            for indk, kkk in enumerate(keys):
                for iii in range(0,self.nSlicesPerRep):
                    tmp.append([kkk[0]+int(np.fix(iii*(self.chunk_size*(1-self.ov)/self.SF))) , kkk[0]+int(np.fix(self.chunk_size/self.SF+iii*(self.chunk_size*(1-self.ov)/self.SF)))])
            tmp = np.array(tmp)

            # Reshuffling
            self.labelsTrain = labels[self.sliceTrain]
            self.n = self.labelsTrain.shape[0]
            self.keysTrain = tmp[self.sliceTrain]

            if self.equalize:
                # TODO: Equalize number of slices per class
                self.keysTrain, self.labelsTrain = equalizeClasses(self.keysTrain, self.labelsTrain)
                self.n = self.labelsTrain.shape[0]

        else:
            ## Create TEST DATASET HERE

            # SLICING HERE
            tmp = []
            for indk, kkk in enumerate(keys):
                for iii in range(0,self.nSlicesPerRep):
                    tmp.append([kkk[0]+int(np.fix(iii*(self.chunk_size*(1-self.ov)/self.SF))) , kkk[0]+int(np.fix(self.chunk_size/self.SF+iii*(self.chunk_size*(1-self.ov)/self.SF)))])
            tmp = np.array(tmp)

            # Reshuffling
            self.labelsTest = labels[slices_to_take]
            self.n = self.labelsTest.shape[0]
            self.keysTest = tmp[slices_to_take]

            if self.equalize:
                # TODO: Equalize number of slices per class
                self.keysTest, self.labelsTest = equalizeClasses(self.keysTest, self.labelsTest)
                self.n = self.labelsTest.shape[0]

        ## 3. Create transform
        transform, target_transform = self.createTransform(train)

        super(MNDataset, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform)

    def download(self):
        isexisting = super(MNDataset, self).download()

    def __len__(self):
        return self.n

    def __getitem__(self, key):

        if self.train:
            target = self.labelsTrain[key]
            key = self.keysTrain[key]
        else:
            target = self.labelsTest[key]
            key = self.keysTest[key]

        data = self.sample(key, T=int(np.fix(self.chunk_size/self.dt*self.fsamp)), ov=int(np.fix(self.chunk_size*self.ov/self.dt*self.fsamp)))

        # if self.batch_counter > 0 & np.mod(self.batch_counter,)

        if self.transform is not None:
            data = self.transform(data)


        if self.target_transform is not None:
            target = self.target_transform(target)


        return data, target

    def sample(self, key, T=300, ov=0):

        times = []
        addr = []
        mmm = 0 # Counter of motor neuron number -> Address, start from 0
        NUMFIRINGS = 0 # Count number total firings in the chunk

        # Run across muscles
        for iii in range(0, len(self.muscles)):
            # Exclude muscles in self.muscle2Excl
            if (np.argwhere(np.array(self.muscle2Excl) == iii + 1)).shape[0] == 0:
                # If contains motor neurons
                if self.data[iii, 0].shape[1] > 0:
                    # Run across motorneurons in the muscle
                    for jjj in range(0, self.data[iii, 0].shape[0]):
                        if (key[0]+self.key_counter*T+T) <= key[1]:
                            # Equivalent of "get_tmad_slice", but we can try also with that function
                            times.extend(self.data[iii, 0][jjj, 0][(self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T))])
                            addr.extend([mmm for _ in range(0, np.sum((self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T))))])
                            NUMFIRINGS += np.sum((self.data[iii, 0][jjj, 0] >= key[0]) & (self.data[iii, 0][jjj, 0] < (key[0]+self.key_counter*T+T)))
                            mmm += 1

        # Remove here times, addr items less than firing thr
        if NUMFIRINGS >= np.fix(self.chunk_size / 1000 * self.thr_firing_excl_slice * self.nMN / 10):
            ADDR = np.array(addr)
            addr = ADDR[np.argsort(times)] #.tolist()
            times = np.sort(times) #.tolist()
            if len(times) > 0 :
                times -= times[0]

        tmad = np.column_stack([times, addr]).astype(int)
        # self.key_counter += 1

        return tmad

    def createTransform(self, train):

        downsampfact = self.fsamp / self.dt

        if train:

            transform = Compose([
                Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
                ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
                ToTensor()])

            target_transform = Compose([Repeat(self.chunk_size), toOneHot(self.nclasses)])
        else:

            transform = Compose([
                Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
                ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
                ToTensor()])

            target_transform = Compose([Repeat(self.chunk_size), toOneHot(self.nclasses)])

        return transform, target_transform

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
    for nc in range(0, max(labels)+1):
        m = np.min([np.sum(labels == nc), m])
    m = int(m)
    kk = np.zeros([0,  keys .shape[1]])
    ll = []
    for nc in range(0, max(labels)+1):
        kk = np.row_stack([kk,  keys[labels == nc][0:m]])
        ll.extend([nc for iii in range(0, m)])
    ll = np.array(ll)
    kk = kk.astype(int)
    return kk, ll

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
    # # Commented (Ctrl+1) by Simone Tanzarella 19/01/2021
    train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
    # # Modified by Simone Tanzarella 19/01/2021
    # # train_dl = torch.utils.data.DataLoader(train_d, shuffle=False, batch_size=batch_size, **dl_kwargs)
    test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

    return train_dl, test_dl
    # return train_d, test_d