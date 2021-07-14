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



def init_LSUV(net, data_batch, mu=0.0, var=1.0):
    '''
    Initialization inspired from Mishkin D and Matas J. All you need is a good init. arXiv:1511.06422 [cs],
February 2016.
    '''
    ##Initialize
    if mu is None:
        mu = 0.0
    if var is None:
        var = 1.0
    with torch.no_grad():
        net.init_parameters(data_batch)
        #def lsuv(net, data_batch):
        for l in net.LIF_layers:
            l.base_layer.bias.data *= 0
            init.orthogonal_(l.base_layer.weight)

            if hasattr(l,'rec_layer'):
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
                print(i,v,m,mus)
                if np.isnan(v) or np.isnan(m):
                    print('Nan encountered during init')
                    mus = -.1
                if np.abs(v-var)>.1:
                    net.LIF_layers[i].base_layer.weight.data /= np.sqrt(v)*np.sqrt(var)
                    ## Won't converge:
                    #if hasattr(net.LIF_layers[i],'rec_layer'):
                    #    net.LIF_layers[i].rec_layer.weight.data /= np.sqrt(v)*np.sqrt(var)
                    done=False
                else:
                    done=True
                    
                if np.abs(m-mu+.1)>.2:
                    net.LIF_layers[i].base_layer.bias.data -= .5*(m-mu) 
                    #if hasattr(net.LIF_layers[i],'rec_layer'):
                    #    net.LIF_layers[i].rec_layer.bias.data -= .5*(m-mu) 
                    done=False
                else:
                    done=True
                alldone*=done
                
                
def init_LSUV_actrate(net, data_batch, act_rate, threshold=0., var=1.0):
    from scipy.stats import norm
    import scipy.optimize
    tgt_mu = scipy.optimize.fmin(lambda loc: (act_rate-(1-norm.cdf(threshold,loc,var)))**2, x0=0.)[0]
    init_LSUV(net, data_batch, mu=tgt_mu, var=var)
    
