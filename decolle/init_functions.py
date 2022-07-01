#!/bin/python
#-----------------------------------------------------------------------------
# File Name : init_functions.py
# Author: Emre Neftci
#
# Creation Date : Fri 26 Feb 2021 11:48:40 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
import torch
import numpy as np

from torch.nn import init



def init_LSUV(net, data_batch, tgt_mu=0.0, tgt_var=1.0):
    '''
    Initialization inspired from Mishkin D and Matas J. All you need is a good init. arXiv:1511.06422 [cs],
February 2016.
    '''
    ##Initialize
    with torch.no_grad():
        net.init_parameters(data_batch)
        #def lsuv(net, data_batch):
        for l in net.LIF_layers:
            if l.base_layer.bias is not None:
                l.base_layer.bias.data *= 0
            init.orthogonal_(l.base_layer.weight)

            if hasattr(l,'rec_layer'):
                if l.rec_layer.bias is not None:
                    l.rec_layer.bias.data *= 0
                init.orthogonal_(l.rec_layer.weight)
        alldone = False
        while not alldone:
            alldone = True
            s,r,u = net.process_output(data_batch)
            for i in range(len(net)):
                v=np.var(u[i][-1].flatten())
                m=np.mean(u[i][-1].flatten())
                mus=np.mean(s[i][-1].flatten())                
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(i,v,m,mus))
                if np.isnan(v) or np.isnan(m):
                    print('Nan encountered during init')
                    done = False
                    raise
                if np.abs(v-tgt_var)>.1:
                    net.LIF_layers[i].base_layer.weight.data /= np.sqrt(np.maximum(v,1e-3))                  
                    net.LIF_layers[i].base_layer.weight.data *= np.sqrt(tgt_var)
                    done=False
                else:
                    done=True
                alldone*=done
                    
                if np.abs(m-tgt_mu)>.2:
                    if net.LIF_layers[i].base_layer.bias is not None:
                        net.LIF_layers[i].base_layer.bias.data -= .5*(m-tgt_mu) 
                    done=False
                else:
                    done=True
                alldone*=done
            if alldone:
                print("Initialization finalized:")
                print("Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(i,v,m,mus))

                
                
def init_LSUV_actrate(net, data_batch, act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    tgt_mu = scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]
    init_LSUV(net, data_batch, tgt_mu=tgt_mu, tgt_var=var)
    
