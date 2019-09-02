#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Author: Emre Neftci
#
# Creation Date : Fri 01 Dec 2017 10:05:17 PM PST
# Last Modified : Sun 29 Jul 2018 01:39:06 PM PDT
#
# Copyright : (c)
# Licence : Apache License, Version 2.0
#-----------------------------------------------------------------------------
import struct
import numpy as np
import scipy.misc
import h5py
import glob
from decolle.events_timeslices import *
import os
dcll_folder = os.path.dirname(__file__)

mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}

class SequenceGenerator(object):
    def __init__(self,
        filename = 'data/dvs_gestures_events.hdf5',
        group = 'train',
        batch_size = 32,
        chunk_size = 500,
        ds = 2,
        size = [2, 64, 64],
        dt = 1000):

        self.group = group
        self.dt = dt
        self.ds = ds
        self.size = size
        f = h5py.File(filename, 'r', swmr=True, libver="latest")
        self.stats = f['stats']
        self.grp1 = f[group]
        self.num_classes = 11
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def reset(self):
        self.i = 0

    def next(self, offset = 0):
        if self.group != 'test':
            dat,lab = next(
                    self.grp1,
                    stats = self.stats,
                    batch_size = self.batch_size,
                    T = self.chunk_size,
                    n_classes = self.num_classes,
                    size = self.size,
                    ds = self.ds,
                    dt = self.dt)
        else:
            dat,lab = next_1ofeach(
                    self.grp1,
                    T = self.chunk_size,
                    n_classes = self.num_classes,
                    size = self.size,
                    ds = self.ds,
                    dt = self.dt,
                    offset = offset)

        return dat, lab

#def find_first(a, i):
#    return np.searchsorted(a,i)

#def find_first(a, tgt):
#    for i,aa in enumerate(a):
#        if aa>tgt:
#            return i
#    return len(a)

def gather_gestures_stats(hdf5_grp):
    from collections import Counter
    labels = []
    for d in hdf5_grp:
        labels += hdf5_grp[d]['labels'][:,0].tolist()
    count = Counter(labels)
    stats = np.array(list(count.values()))
    stats = stats/ stats.sum()
    return stats




def gather_aedat(directory, start_id, end_id, filename_prefix = 'user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))
    import glob
    fns = []
    for i in range(start_id,end_id):
        search_mask = directory+'/'+filename_prefix+"{0:02d}".format(i)+'*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out)>0:
            fns+=glob_out
    return fns

def aedat_to_events(filename):
    label_filename = filename[:-6] +'_labels.csv'
    labels = np.loadtxt(label_filename, skiprows=1, delimiter=',',dtype='uint32')
    events=[]
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True:
            data_ev_head = f.read(28)
            if len(data_ev_head)==0: break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if(eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber*eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1,2)

                x = (event_bytes[:,0] >> 17) & 0x00001FFF
                y = (event_bytes[:,0] >> 2 ) & 0x00001FFF
                p = (event_bytes[:,0] >> 1 ) & 0x00000001
                t = event_bytes[:,1]
                events.append([t,x,y,p])

            else:
                f.read(eventnumber*eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    clipped_events = np.zeros([4,0],'uint32')
    for l in labels:
        start = np.searchsorted(events[0,:], l[1])
        end = np.searchsorted(events[0,:], l[2])
        clipped_events = np.column_stack([clipped_events,events[:,start:end]])
    return clipped_events.T, labels

#def sort_events_bylabel(events, labels, n_classes = 11, t_cont = None):
#    if t_cont is None:
#        t_cont = np.zeros(n_classes, 'int')
#    T = events[0, -1]
#    idx = events[3,:] == pol
#    events = events[:,idx]
#    t0 = events[0, 0]
#    events[0,:] -= t0
#
#    start_idx = 0
#    end_idx = 0
#
#    labels[:,1] =(labels[:,1]-t0)//deltat
#    labels[:,2] =(labels[:,2]-t0)//deltat
#
#    sorted_events = [np.zeros([4,0]) for i in range(n_classes)]
#
#    for label, start, end in labels:
#        sorted_events[label] = np.concatenate([sorted_events[label],events[:,:,:,start:end]])
#
#    return sorted_events
def compute_start_time(labels,pad):
    l0 = np.arange(len(labels[:,0]), dtype='int')
    np.random.shuffle(l0)
    label = labels[l0[0],0]
    tbegin = labels[l0[0],1]
    tend = labels[l0[0],2]-pad
    start_time = np.random.randint(tbegin, tend)
    return start_time, label

def next(hdf5_group, stats, batch_size = 32, T = 500, n_classes = 11, ds = 2, size = [2, 64, 64], dt = 1000):
    batch = np.empty([batch_size,T]+size, dtype='float')
    batch_idx = np.arange(len(hdf5_group), dtype='int')
    np.random.shuffle(batch_idx)
    batch_idx = batch_idx[:batch_size]
    batch_idx_l = np.empty(batch_size, dtype= 'int')
    for i, b in (enumerate(batch_idx)):
        dset = hdf5_group[str(b)]
        labels = dset['labels'].value
        cand_batch = -1
        while cand_batch is -1: #catches some mislabeled data
            start_time, label = compute_start_time(labels, pad = 2*T*dt)
            batch_idx_l[i] = label-1
            #print(str(i),str(b),mapping[batch_idx_l[i]], start_time)
            cand_batch = get_event_slice(dset['time'].value, dset['data'], start_time, T, ds=ds, size=size, dt=dt)
        batch[i] = cand_batch

    #print(batch_idx_l)
    return batch, one_hot(batch_idx_l, n_classes).astype('float')

def next_1ofeach(hdf5_group, T = 500, n_classes = 11, ds = 2, size = [2, 64, 64], dt = 1000, offset = 0):
    batch_1of_each = {k:range(len(v['labels'].value)) for k,v in hdf5_group.items()}
    batch_size = np.sum([len(v) for v in batch_1of_each.values()]).astype(int)
    batch = np.empty([batch_size,T]+size, dtype='float')
    batch_idx_l = np.empty(batch_size, dtype= 'int')
    i = 0
    for b,v in batch_1of_each.items():
        for l0 in v:
            dset = hdf5_group[str(b)]
            labels = dset['labels'].value
            label = labels[l0,0]
            batch_idx_l[i] = label-1
            start_time = labels[l0,1] + offset*dt
            #print(str(i),str(b),mapping[batch_idx_l[i]], start_time)
            batch[i] = get_event_slice(dset['time'].value, dset['data'], start_time, T, ds=ds, size=size, dt=dt)
            i += 1
    return batch, one_hot(batch_idx_l, n_classes).astype('float')


#def get_time_slice(labels, label, T):
#    idxs = np.where([labels[:,0] == label])[1]
#    np.random.shuffle(idxs)
#    idx = idxs[0]
#    start_time = labels[idx][1]
#
#    return start_time

def get_event_slice(times, addrs, start_time, T, size = [128,128], ds = 1, dt = 1000):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time+T*dt)+idx_beg
        return chunk_evs_pol(times[idx_beg:idx_end], addrs[idx_beg:idx_end], deltat=dt, chunk_size=T, size = size, ds_w=ds, ds_h=ds)
    except IndexError:
        print("Empty batch found, returning -1")
        return -1

def create_events_hdf5(hdf5_filename):
    fns_train = gather_aedat(hdf5_filename,1,24)
    fns_test = gather_aedat (hdf5_filename,24,30)

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        print("processing training data...")
        key = 0
        train_grp = f.create_group('train')
        for file_d in fns_train:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = train_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        print("processing testing data...")
        key = 0
        test_grp = f.create_group('test')
        for file_d in fns_test:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = test_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        stats =  gather_gestures_stats(train_grp)
        f.create_dataset('stats',stats.shape, dtype = stats.dtype)
        f['stats'][:] = stats

def create_data(filename = 'data/dvs_gestures_events.hdf5', batch_size = 64 , chunk_size_train = 500, chunk_size_test = 500, size = [2, 32, 32], ds = 4, dt = 1000):
    if not os.path.isfile(filename):
        print("File {} does not exist: converting DvsGesture to h5file".format(filename))
        create_events_hdf5(filename)
    else:
        print("File {} exists: not re-converting DvsGesture".format(filename))

    strain = SequenceGenerator(filename, group='train', batch_size = batch_size, chunk_size = chunk_size_train, size = size, ds = ds, dt= dt)
    stest = SequenceGenerator(filename, group='test', batch_size = batch_size, chunk_size = chunk_size_test, size = size, ds = ds, dt= dt)
    return strain, stest

def plot_gestures_imshow(images, labels, nim=11, avg=50, do1h = True, transpose=False):
    import pylab as plt
    plt.figure(figsize = [nim+2,16])
    import matplotlib.gridspec as gridspec
    if not transpose:
        gs = gridspec.GridSpec(images.shape[1]//avg, nim)
    else:
        gs = gridspec.GridSpec(nim, images.shape[1]//avg)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=.0, hspace=.04)
    if do1h:
        categories = labels.argmax(axis=1)
    else:
        categories = labels
    s=[]
    for j in range(nim):
         for i in range(images.shape[1]//avg):
             if not transpose:
                 ax = plt.subplot(gs[i, j])
             else:
                 ax = plt.subplot(gs[j, i])
             plt.imshow(images[j,i*avg:(i*avg+avg),0,:,:].sum(axis=0).T)
             plt.xticks([])
             if i==0:  plt.title(mapping[labels[0,j].argmax()], fontsize=10)
             plt.yticks([])
             plt.gray()
         s.append(images[j].sum())
    print(s)
    #return images,labels



    #pass

if __name__ == '__main__':
    gen_train, gen_test = create_data(chunk_size_train=10, chunk_size_test=20, batch_size=50, size=[2, 128, 128], dt = 50000, ds =1)
    data_batch, target_batch = gen_train.next()
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    im = None
    for b in tqdm(data_batch):
        for d in b:
            if im is None:
                im = plt.imshow(d[0] - d[1])
            else:
                im.set_data(d[0] - d[1])
                im.autoscale()
            plt.pause(0.01)
            plt.draw()
        pass
