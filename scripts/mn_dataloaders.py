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


mapping = {0: 'index_flx',
           1: 'index_ext',
           2: 'middle_flx',
           3: 'middle_ext',
           4: 'ring_flx',
           5: 'little_flx',
           6: 'little_ext',
           7: 'thumb_flx',
           8: 'thumb_ext'}

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
            muscle_to_exclude=[],
            dt=1000):

        self.n = 0
        self.nclasses = self.num_classes = 9
        self.download_and_create = download_and_create
        self.root = root
        self.train = train
        self.dt = dt
        self.chunk_size = chunk_size
        self.ov=overlap_perc
        self.key_counter = 0
        self.batch_counter = 0
        self.muscle2Excl = muscle_to_exclude

        ## 1. LOAD THE FILE
        matfile = scipy.io.loadmat(root)
        self.data = matfile['e']['mu'][0, 0][0, 0][1]
        self.muscles = []
        for iii in range(0, matfile['e']['muscles'][0, 0].shape[0]):
            # TODO: : Insert here the condition for muscles to exclude
            if (np.argwhere(np.array(self.muscle2Excl) == iii + 1)).shape[0] == 0:
                self.muscles.append(matfile['e']['muscles'][0, 0][iii, 0][0])

        self.fsamp = int(matfile['e']['fsamp'])

        print('Number of considered muscles : ' + str(self.data.shape[0]))
        print('Number of motor neurons per muscle :')
        self.nMN = 0
        for iii in range(0, self.data.shape[0]):
            # TODO: : Insert here the condition for muscles to exclude
            if (np.argwhere(np.array(self.muscle2Excl)==iii+1)).shape[0] == 0:
                if self.data[iii, 0].shape[1] > 0:
                    self.nMN += self.data[iii, 0].shape[0]
                    print(self.muscles[iii] + ', ' + str(self.data[iii, 0].shape[0]) + ' MNs')
                else:
                    print(self.muscles[iii] + ', ' + str(self.data[iii, 0].shape[1]) + ' MNs')

        ## 2. Create transform
        transform, target_transform = self.createTransform(train)

        super(MNDataset, self).__init__(
            root,
            transform=transform,
            target_transform=target_transform)

        ## 3. Extract here labels and keys
        # TODO : dynamic assignation test and train portions
        self.labelsTrain = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] == 0, 2]-1
        self.labelsTest = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] > 0, 2]-1
        self.keysTrain = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] == 0, 0:2]
        self.keysTest = matfile['e']['Keys'][0, 0][matfile['e']['Keys'][0, 0][:, 3] > 0, 0:2]

        self.repWidth = int(np.fix(np.mean(np.diff(self.keysTrain, 1, 1))))
        # self.nSlicesPerRep = np.fix(self.repWidth/(self.chunk_size/1000*self.fsamp))
        self.SF = self.dt /self.fsamp
        self.nSlicesPerRep = int(np.fix((self.repWidth-(self.chunk_size*self.ov)/self.SF) / ((self.chunk_size*(1-self.ov))/self.SF)))

        if train:
            self.n = int(self.nSlicesPerRep*self.labelsTrain.shape[0])
            self.labelsTrain = np.repeat(self.labelsTrain, int(self.nSlicesPerRep))
            # self.keysTrain = np.repeat(self.keysTrain, int(self.nSlicesPerRep),axis=0)
            tmp = []
            for kkk in self.keysTrain:
                for iii in range(0,self.nSlicesPerRep):
                    tmp.append([kkk[0]+int(np.fix(iii*(self.chunk_size*(1-self.ov)/self.SF))) , kkk[0]+int(np.fix(self.chunk_size/self.SF+iii*(self.chunk_size*(1-self.ov)/self.SF)))])
            self.keysTrain = tmp
        else:
            self.n = int(self.nSlicesPerRep * self.labelsTest.shape[0])
            self.labelsTest = np.repeat(self.labelsTest, int(self.nSlicesPerRep))
            # self.keysTest = np.repeat(self.keysTest, int(self.nSlicesPerRep), axis=0)
            tmp = []
            for kkk in self.keysTest:
                for iii in range(0,self.nSlicesPerRep):
                    tmp.append([kkk[0]+int(np.fix(iii*(self.chunk_size*(1-self.ov)/self.SF))) , kkk[0]+int(np.fix(self.chunk_size/self.SF+iii*(self.chunk_size*(1-self.ov)/self.SF)))])
            self.keysTest = tmp
        # # f = h5py.File(root, 'r', swmr=True, libver="latest")
        #
        # # Commented (Ctrl+1) by Simone Tanzarella 19/01/2021
        # with h5py.File(root, 'r', swmr=True, libver="latest") as f:
        #     try:
        #         if train:
        #             self.n = f['extra'].attrs['Ntrain']
        #             self.keys = f['extra']['train_keys'][()]
        #             self.keys_by_label = f['extra']['train_keys_by_label'][()]
        #         else:
        #             self.n = f['extra'].attrs['Ntest']
        #             self.keys = f['extra']['test_keys'][()]
        #             self.keys_by_label = f['extra']['test_keys_by_label'][()]
        #             self.keys_by_label[:, :] -= self.keys_by_label[0, 0]  # normalize
        #     except AttributeError:
        #         print(
        #             'Attribute not found in hdf5 file. You may be using an old hdf5 build. Delete {0} and run again'.format(
        #                 root))
        #         raise

    def download(self):
        isexisting = super(MNDataset, self).download()

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        # HERE THE CALL FUNCTION 'ITER' applyied to this object will iter key counter
        # # Important to open and close in getitem to enable num_workers>0
        # with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
        #     if self.train:
        #         key = f['extra']['train_keys'][key]
        #     else:
        #         key = f['extra']['test_keys'][key]
        #     data, target = sample(
        #         f,
        #         key,
        #         T=self.chunk_size * self.dt)

        # TODO : debug this code, do you really need "transform"? How to adapt it?
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

        # dset = hdf5_file['data'][str(key)]
        # label = dset['labels'][()]
        # tend = dset['times'][-1]
        # start_time = 0
        # ha = dset['times'][()]
        #
        # tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T * 1000)
        # tmad[:, 0] -= tmad[0, 0]
        # print(key)
        times = []
        addr = []
        mmm = 0 # Counter of motor neuron number -> Address, start from 0

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
                            mmm += 1

            # else:
            #     print(self.muscles[iii]+' '+'empty')
        ADDR = np.array(addr)
        addr = ADDR[np.argsort(times)] #.tolist()
        times = np.sort(times) #.tolist()
        if len(times) > 0 :
            times -= times[0]
        tmad = np.column_stack([times, addr]).astype(int)
        # self.key_counter += 1

        return tmad

    def createTransform(self, train):

        # size = [self.nMN], the total number of motor neurons

        # Since we express T in milliseconds, the downsampling factor is the sample frequency,
        # i.e. the number of samples acquired in a second of the original EMG signal (then decomposed
        # to find motor neuron firing times), divided by 1000 (dt) since we would need to express in millisecond the firing times,
        # instead of seconds,
        # which would be if we don't divide the downsampling factor per 1000 (i.e. re-oversample of 1000).
        downsampfact = self.fsamp / self.dt

        if train:

            transform = Compose([
                Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
                ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
                ToTensor()])

            target_transform = Compose([Repeat(self.chunk_size), toOneHot(len(mapping))])
        else:

            transform = Compose([
                Downsample(factor=[downsampfact, 1]),# , 1, 1]), Address has only one coordinate,i.e. the MN number
                ToCountFrameMN(T=self.chunk_size , size=[self.nMN]), #,])
                ToTensor()])

            target_transform = Compose([Repeat(self.chunk_size), toOneHot(len(mapping))])

        return transform, target_transform

    def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, shuffle=True):
        """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.
        Args:
            X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
            y: The labels
        """

        labels_ = np.array(y, dtype=np.float)
        number_of_batches = len(X) // batch_size
        sample_index = np.arange(len(X))

        # compute discrete firing times
        tau_eff = 20e-3 / time_step
        firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.float)
        unit_numbers = np.arange(nb_units[1] * nb_units[2])
        # unit_numbers_2 = np.arange(nb_units[2])

        if shuffle:
            np.random.shuffle(sample_index)

        # total_batch_count = 0
        counter = 0
        while counter < number_of_batches:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

            coo = [[] for i in range(5)]
            for bc, idx in enumerate(batch_index):
                for n_mat in range(0, nb_units[0]):
                    c = firing_times[idx] < nb_steps
                    times, units = firing_times[idx][c], unit_numbers[c]

                    cols = np.fix((units + 1) / nb_units[1] - 1).astype(int)  # cols
                    rows = (np.mod((units), nb_units[1])).astype(int)  # rows

                    batch = [bc for _ in range(len(times))]
                    matrices = [n_mat for _ in range(len(times))]
                    coo[0].extend(batch)
                    coo[1].extend(times)
                    coo[2].extend(matrices)
                    coo[3].extend(rows)
                    coo[4].extend(cols)

            i = torch.LongTensor(coo).to(device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
                [batch_size, nb_steps, nb_units[0], nb_units[1], nb_units[2]])).to(device)
            # y_batch = torch.tensor(labels_[batch_index],device=device)
            LL = []
            for lab in labels_[batch_index]:
                LL.append([[lab]])
            y_batch = torch.tensor(np.repeat(np.array(LL), 300, axis=1), device=device)

            yield X_batch.detach().cpu().to_dense().numpy(), y_batch.detach().cpu().numpy()

            counter += 1


def create_datasets(
        root='data/motoneurons/MNDS_KaJu.mat',
        batch_size=72,
        chunk_size_train=300,
        chunk_size_test=300,
        overlap_size_train_perc = 0,
        overlap_size_test_perc = 0,
        muscle_to_exclude=[],
        ds=1,
        dt=1000):

    train_ds = MNDataset(root, train=True,
                             chunk_size=chunk_size_train,
                             overlap_perc=overlap_size_train_perc,
                             muscle_to_exclude=muscle_to_exclude,
                             dt=dt)

    test_ds = MNDataset(root, train=False,
                            chunk_size=chunk_size_test,
                            overlap_perc=overlap_size_test_perc,
                            muscle_to_exclude=muscle_to_exclude,
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
        muscle_to_exclude = [],
        ds=1,
        dt=1000,
        **dl_kwargs):
    train_d, test_d = create_datasets(
        root='data/motoneurons/MNDS_KaJu.mat',
        batch_size=batch_size,
        chunk_size_train=chunk_size_train,
        chunk_size_test=chunk_size_test,
        overlap_size_train_perc=overlap_size_train_perc,
        overlap_size_test_perc=overlap_size_test_perc,
        muscle_to_exclude=muscle_to_exclude,
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