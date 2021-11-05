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
import scipy.io as io
# from scipy import signal
import random
import inspect
import os


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
            thr_firing_excl_slice=0,
            dt=1000,
            class_to_copy=[],
            slices_for_train_in=[]):

        if not(train):
            self.clone_class(class_to_copy)
            self.train = train
        else:
            # Inputs
            self.n = 0
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
            self.slices_for_train_in = slices_for_train_in
            
            # Generated
            self.slices_for_train = {'hyper_param': [], 'data_select': []}
            self.equalize = False
            self.slices_to_take = []


            ## 1. LOAD THE FILE
            self.load_file()
            
            ## 2. Check whether classes are equally represented, if not we'll make them equal
            self.check_equal_repr_classes()

            ## 3. Select muscles
            self.select_muscles()

            ## 4. Exclude classes at this level
            
            self.select_classes()

            # Set number of slices to cut
            self.slices_to_cut()

            # SLICING WHOLE ORIGINAL DATA HERE
            self.slice_whole_data()

            if type(slices_for_train_in) is dict:
                # Take a subset of slices( different between
                # hyper-parameter tuning and data-selection tuning)
                self.define_slices_to_take()

            # Define type of training
            if type(slices_for_train_in) is dict:
                if (self.slices_for_train_in['type_train'].__contains__('hyp')) | (self.slices_for_train_in['type_train'].__contains__('par')):
                    # Hyper-parameter tuning
                    self.reshuffle_slices(self.slices_for_train['hyper_param'])
                else:
                    # Data selection (segmentation, muscles, classes, contraction types)
                    # tuning
                    self.reshuffle_slices(self.slices_for_train['data_select'])

            # Prepare reshuffling train/test here!!!
            self.divide_slices_train_test()

        if self.train:
            ## Create TRAIN DATASET HERE
            # Reshuffling
            self.reshuffle_slices(self.sliceTrain)
        else:
            ## Create TEST DATASET HERE
            # Reshuffling
            self.reshuffle_slices(self.sliceTest)

        if self.equalize:
            # TODO: Equalize number of slices per class
            self.keys_taken, self.targets, self.slices_to_take = equalizeClasses(self.keys_taken, self.targets)
        self.n = self.targets.shape[0]

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

        target = self.labels[key]
        key = self.keys[key]

        data = self.sample(key, T=int(np.fix(self.chunk_size/self.dt*self.fsamp)), ov=int(np.fix(self.chunk_size*self.ov/self.dt*self.fsamp)))

        # if self.batch_counter > 0 & np.mod(self.batch_counter,)

        if self.transform is not None:
            data = self.transform(data)


        if self.target_transform is not None:
            target = self.target_transform(target)


        return data, target

    def load_file(self):
        # Load file with data
        self.subj = self.root[(self.root.find('MNDS_') + 5):(self.root.find('MNDS_') + 9)]
        matfile = scipy.io.loadmat(self.root)
        # We take all concatenate motor units and not mu0 = matfile['e']['mu'][0, 0][0, 0]['dataOnlyCommon']
        self.data = matfile['e']['mu'][0, 0][0, 0]['dataOnlyCommon']
        self.fsamp = int(matfile['e']['fsamp'])
        self.list_all_musc = matfile['e']['muscles'][0, 0]
        KEYS = matfile['e']['Keys'][0, 0]

        self.labels = KEYS[:, 2] - 1*(min(KEYS[:, 2]) == 1)
        self.keys = KEYS[:, 0:2]


    def check_equal_repr_classes(self):
        m = []
        for nc in range(0, max(self.labels)+1):
            m.append(np.sum(np.diff(self.keys[self.labels == nc, :], axis=1)))
        if min(m) < max(m):
            self.equalize = True

    def select_muscles(self):
        print('Number of considered muscles : ' + str(self.data.shape[0]-len(self.muscle2Excl)))
        print('Number of motor neurons per muscle :')

        self.muscles = []
        self.nMN = 0
        data_tmp = []
        for iii in range(0, self.list_all_musc.shape[0]):
            # Exclude muscles in self.muscle2Excl
            if (np.argwhere(np.array(self.muscle2Excl) == iii + 1)).shape[0] == 0:
                self.muscles.append(self.list_all_musc[iii, 0][0])
                data_tmp.append(self.data[iii, 0])
                if self.data[iii, 0].shape[1] > 0:
                    self.nMN += self.data[iii, 0].shape[0]
                print(self.list_all_musc[iii, 0][0] + ', ' + str(self.data[iii, 0].shape[0]) + ' MNs')
        # From "list" to the original format
        self.data = np.expand_dims(np.array(data_tmp), axis=1)

    def select_classes(self):
        # CLASS SELECTION HERE

        # Number of classes
        self.nclasses = max(self.labels) + 1*(min(self.labels) == 0)
        
        self.keys = np.array(self.keys)
        if type(self.class_to_include) is list:
            if not(len(self.class_to_include) == 0):
                self.nclasses = 0
                # Inizialize the new labels and intervals
                l = []
                k = np.zeros([0,self.keys.shape[1]])
                for nc in self.class_to_include:
                    # If the nc class is included
                    l.extend([int(self.nclasses) for iii in range(0, np.sum([self.labels == nc]))])
                    k = np.row_stack([k, self.keys[self.labels == nc,: ]])
                    self.nclasses+=1
                # New labels and intervals
                self.labels = np.array(l)
                self.keys = k.astype(int)

    def slices_to_cut(self):
        self.repWidth = int(np.fix(np.mean(np.diff(self.keys, 1, 1))))
        self.SF = self.dt /self.fsamp
        self.nSlicesPerRep = int(np.fix((self.repWidth-(self.chunk_size*self.ov)/self.SF) / ((self.chunk_size*(1-self.ov))/self.SF)))

        self.n = int(self.nSlicesPerRep * self.labels.shape[0])
        self.labels = np.repeat(self.labels, int(self.nSlicesPerRep))

    def define_slices_to_take(self):
        slice_for_train_path = os.getcwd() + '/' + self.slices_for_train_in['folder'] + '/'
        slice_for_train_path = slice_for_train_path.replace('\\','//')
        if not os.path.isdir(slice_for_train_path):
            # If there is not a folder to take the .mat file for selecting the slices to take 
            # on the global dataset, make the folder
            os.mkdir(slice_for_train_path)
        if not os.path.isfile(slice_for_train_path + 'slices_for_train_'+ self.subj +'_'+ str(self.nclasses) + '_'+self.root[-16:-4]+ '.mat'):
            # Create here the slice division
            select, hyper = divide_slices(slices_to_take=[],perc_test_norm=self.slices_for_train_in['perc_hyperpar'],n=self.n)
            self.slices_for_train['hyper_param'] = hyper
            self.slices_for_train['data_select'] = select
            io.savemat(slice_for_train_path + 'slices_for_train_'+ self.subj +'_'+ str(self.nclasses) + '_'+self.root[-16:-4]+ '.mat',   self.slices_for_train)
        else:
            data = io.loadmat(slice_for_train_path + 'slices_for_train_'+ self.subj +'_'+ str(self.nclasses) + '_'+self.root[-16:-4]+ '.mat')
            self.slices_for_train['hyper_param'] = np.squeeze( data['hyper_param'], axis = 0 )
            self.slices_for_train['data_select'] = np.squeeze( data['data_select'], axis = 0 )

    def divide_slices_train_test(self):
        self.sliceTrain, self.sliceTest = divide_slices(self.slices_to_take,self.perc_test_norm,self.n)
        

    def slice_whole_data(self):
        tmp = []
        for indk, kkk in enumerate(self.keys):
            for iii in range(0,self.nSlicesPerRep):
                tmp.append([kkk[0]+int(np.fix(iii*(self.chunk_size*(1-self.ov)/self.SF))) , kkk[0]+int(np.fix(self.chunk_size/self.SF+iii*(self.chunk_size*(1-self.ov)/self.SF)))])
        self.keys = np.array(tmp)

    def reshuffle_slices(self,slice_to_take=[]):
        if len(slice_to_take) > 0:
            self.targets = self.labels[slice_to_take]
            self.n = self.targets.shape[0]
            self.keys_taken = self.keys[slice_to_take]

    def clone_class(self,cl2cl):
        self.n = cl2cl.n
        self.download_and_create = cl2cl.download_and_create
        self.root = cl2cl.root
        # NEVER DO THIS : self.train = cl2cl.train !!!!!
        self.dt = cl2cl.dt
        self.chunk_size = cl2cl.chunk_size
        self.ov=cl2cl.ov
        self.key_counter = cl2cl.key_counter
        self.batch_counter = cl2cl.batch_counter
        self.muscle2Excl = cl2cl.muscle2Excl
        self.perc_test_norm = cl2cl.perc_test_norm
        self.class_to_include = cl2cl.class_to_include
        self.thr_firing_excl_slice = cl2cl.thr_firing_excl_slice
        self.equalize = cl2cl.equalize
        self.slices_to_take = cl2cl.slices_to_take
        self.data = cl2cl.data
        self.fsamp = cl2cl.fsamp
        self.list_all_musc = cl2cl.list_all_musc
        self.labels = cl2cl.labels
        self.keys = cl2cl.keys
        self.muscles = cl2cl.muscles
        self.nMN = cl2cl.nMN
        self.nclasses = cl2cl.nclasses
        self.repWidth = cl2cl.repWidth
        self.SF = cl2cl.SF
        self.nSlicesPerRep = cl2cl.nSlicesPerRep
        self.sliceTrain = cl2cl.sliceTrain
        self.sliceTest = cl2cl.sliceTest
        self.slices_for_train = cl2cl.slices_for_train
        self.slices_to_take = []


    def sample(self, key, T=300, ov=0):

        times = []
        addr = []
        mmm = 0 # Counter of motor neuron number -> Address, start from 0
        NUMFIRINGS = 0 # Count number total firings in the chunk

        # Run across muscles
        for iii in range(0, self.data.shape[0]):
            
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


#----------------------------------------------------------------------
# Functions
#----------------------------------------------------------------------

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
    slices_to_take = np.arange(0, len(ll))
    random.shuffle(slices_to_take)
    ll = np.array(ll)
    kk = kk.astype(int)
    ll = ll[slices_to_take]
    kk = kk[slices_to_take]
    
    return kk, ll, slices_to_take

def divide_slices(slices_to_take=[],perc_test_norm=0,n=0):
    sliceA = np.array([])
    sliceB = np.array([])
    if not (type(slices_to_take) is np.ndarray):
        slices_to_take = np.arange(0, n)
    elif slices_to_take.shape[0] == 0:
        slices_to_take = np.arange(0, n)
    random.shuffle(slices_to_take)
    if (type(n) is int):
        if n > 0:
            sliceA = slices_to_take[0:int(np.fix((1-perc_test_norm)*n))]
            sliceB = slices_to_take[int(np.fix((1 - perc_test_norm) * n)):]
    return sliceA, sliceB

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
        dt=1000,
        slices_for_train_in=[]):

    train_ds = MNDataset(root, train=True,
                            chunk_size=chunk_size_train,
                            overlap_perc=overlap_size_train_perc,
                            perc_test_norm=perc_test_norm,
                            muscle_to_exclude=muscle_to_exclude,
                            class_to_include=class_to_include,
                            thr_firing_excl_slice=thr_firing_excl_slice,
                            dt=dt,
                            class_to_copy=[],
                            slices_for_train_in=slices_for_train_in
                             )

    test_ds = MNDataset(root, train=False,
                            class_to_copy=train_ds)

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
        slices_for_train_in=[],
        **dl_kwargs):
        
    train_d, test_d = create_datasets(
        root=root,
        batch_size=batch_size,
        chunk_size_train=chunk_size_train,
        chunk_size_test=chunk_size_train,
        overlap_size_train_perc=overlap_size_train_perc,
        overlap_size_test_perc=overlap_size_train_perc,
        perc_test_norm=perc_test_norm,
        slices_for_train_in=slices_for_train_in,
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