#!/bin/python
#-----------------------------------------------------------------------------
# File Name : boxlif.py
# Author: Emre Neftci
#
# Creation Date : Wed 03 Jun 2020 03:26:19 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from ..base_model import *

class BoxLIFLayer(LIFLayerVariableTau):
    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, cutoff=5000 , do_detach=True, random_tau = True):
        super(BoxLIFLayer, self).__init__(layer, alpha, alpharp, wrp, beta, deltat, do_detach = do_detach, random_tau = random_tau)
        self.cutoff = cutoff
        
    def forward(self, Sin_t):
        mult_fact = (1-self.beta)*(1-self.alpha)*150000
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + Sin_t
        P = self.alpha * state.P + state.Q*mult_fact  # TODO check with Emre: Q or state.Q?
        R = self.alpharp * state.R - state.S * self.wrp
        Pc = (P>self.cutoff).type(P.dtype)*self.cutoff*3 #3 is a factor that scales the gradient pulse. 
        WPc = self.base_layer(Pc)
        WP = self.base_layer(P)
        WPd = (WP-WPc).detach()+WPc
        U = WPd + R
        S = smooth_step(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def __repr__(self):
        return 'BoxLIFLayer cutoff {0}'.format(self.cutoff)
    
    
   
