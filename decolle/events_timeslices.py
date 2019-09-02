#!/bin/python
# -----------------------------------------------------------------------------
# File Name : events_timeslices.py
# Author: Emre Neftci
#
# Creation Date : Thu 16 May 2019 02:13:09 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------


from __future__ import print_function
import bisect
import numpy as np
# from scipy.sparse import coo_matrix as sparse_matrix

def expand_targets(targets, T=500, burnin=0):
    y = np.tile(targets.copy(), [T, 1, 1])
    y[:burnin] = 0
    return y


def one_hot(mbt, num_classes):
    out = np.zeros([mbt.shape[0], num_classes])
    out[np.arange(mbt.shape[0], dtype='int'), mbt.astype('int')] = 1
    return out


def find_first(a, tgt):
    return bisect.bisect_left(a, tgt)


def cast_evs(evs):
    ts = (evs[:, 0] * 1e6).astype('uint64')
    ad = (evs[:, 1:]).astype('uint64')
    return ts, ad

# def get_binary_frame(evs, size = (346,260), ds=1):
#     tr = sparse_matrix((2*evs[:,3]-1,(evs[:,1]//ds,evs[:,2]//ds)), dtype=np.int8, shape=size)
#     return tr.toarray()

def get_subsampled_coordinates(evs, ds_h, ds_w):
    x_coords = evs[:, 1] // ds_w
    y_coords = evs[:, 2] // ds_h
    if x_coords.dtype != np.int:
        x_coords = x_coords.astype(int)
    if y_coords.dtype != np.int:
        y_coords = y_coords.astype(int)
    return x_coords, y_coords


def get_binary_frame_np(arr, evs, ds_w=1, ds_h=1):
    x_coords, y_coords = get_subsampled_coordinates(evs, ds_h, ds_w)
    arr[x_coords, y_coords] = 2 * evs[:, 3] - 1


def get_binary_frame(arr, evs, ds_w=1, ds_h=1):
    x_coords, y_coords = get_subsampled_coordinates(evs, ds_h, ds_w)
    arr[x_coords, y_coords] = 1


def get_event_timeslice(device, Deltat=1000 * 50):
    # flush
    device.get_event()
    t = -1
    evs = []
    while t < Deltat:
        evs_frame_tmp = device.get_event()[0]
        if evs_frame_tmp is None:
            continue
        evs.append(evs_frame_tmp)
        if t == -1:
            t0 = evs_frame_tmp[0, 0]
        try:
            t = (evs_frame_tmp[-1, 0] - t0)
        except TypeError:
            continue

    evs = np.row_stack(evs)
    idx_end = np.searchsorted(evs[:, 0], t0 + Deltat)
    evs_frame = evs[:idx_end]
    evs_frame[:, 0] = -evs_frame[-1, 0] + evs_frame[:, 0]
    # print(evs_frame[0, 0], evs_frame[-1, 0])
    return evs_frame


def get_time_surface(evs, invtau=1e-6, size=(346, 260, 2)):
    tr = np.zeros(size, 'int64') - np.inf

    for ev in evs:
        tr[ev[2], ev[1], ev[3]] = ev[0]

    a = np.exp(tr[:, :, 0] * invtau) - np.exp(tr[:, :, 1] * invtau)

    return a


def chunk_evs(evs, deltat=1000, chunk_size=500, size=[304, 240], ds_w=1, ds_h=1):
    t_start = evs[0, 0]
    ts = range(t_start + chunk_size, t_start + chunk_size * deltat, deltat)
    chunks = np.zeros([len(ts)] + size, dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(evs[idx_end:, 0], t)
        if idx_end > idx_start:
            get_binary_frame_np(chunks[i, ...], evs[idx_start:idx_end], ds_h=ds_h, ds_w=ds_w)
        idx_start = idx_end
    return chunks


def chunk_evs_pol(times, addrs, deltat=1000, chunk_size=500, size=[2, 304, 240], ds_w=1, ds_h=1):
    t_start = times[0]
    ts = range(t_start, t_start + chunk_size * deltat, deltat)
    chunks = np.zeros([len(ts)] + size, dtype='int8')
    idx_start = 0
    idx_end = 0
    for i, t in enumerate(ts):
        idx_end += find_first(times[idx_end:], t)
        if idx_end > idx_start:
            ee = addrs[idx_start:idx_end]
            pol, x, y = ee[:, 2], (ee[:, 0] // ds_w).astype(np.int), (ee[:, 1] // ds_h).astype(np.int)
            np.add.at(chunks, (i, pol, x, y), 1)
        idx_start = idx_end
    return chunks


if __name__ == "__main__":
    import h5py

    dataset = h5py.File('/home/eneftci_local/Projects/share/data/massiset/massiset_sparse.hdf5', 'r')
    evs = dataset.get('backpack')['data_train'].value
    cevs = chunk_evs(evs, chunk_size=500, deltat=1000, size=[304 // 4, 240 // 4], ds_w=4, ds_h=4)
