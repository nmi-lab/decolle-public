#!/bin/python
#-----------------------------------------------------------------------------
# File Name : lenet_error_model_errotriggered.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from decolle.base_model import *
from .utils import *
from decolle.experimental.boxlif import BoxLIFLayer
from decolle.lenet_decolle_model import LenetDECOLLE
from decolle.lenet_decolle_model_fa import LenetDECOLLEFA
from simple_pid import PID

class BipolarEventEncode(torch.autograd.Function):
    @staticmethod
    def forward(context, input, theta, err_count, count):
        context.save_for_backward(theta, err_count, count)
        return input
    
    @staticmethod
    def backward(context, grad_output):
        theta, err_count, count = context.saved_tensors

        # all of the logic of FA resides in this one line
        # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
        grad_output = grad_output.clone()
        grad_output[grad_output>0] = grad_output[grad_output>0]//theta
        grad_output[grad_output<0] = (-((-grad_output[grad_output<0])//theta))
        #g = torch.sign(g)
        err_count += torch.abs(grad_output).mean().cpu()
        #if err_count !=0: raise
        count += 1
        return grad_output*theta, None, None, None
    
bipolar_grad = BipolarEventEncode().apply

class ErrorLayerBipolar(nn.Module): 
    def __init__(self, shape, init_theta=1e-5, set_point = 5):
        super(ErrorLayerBipolar, self).__init__()
        self.theta  = torch.nn.Parameter(torch.ones(shape)*init_theta, requires_grad = False)
        self.err_count = torch.zeros(shape)
        self.t_count = torch.tensor(0)
        self.relu = nn.ReLU()
        self.set_point = set_point
        self.pid = PID(.5e-6, .0, .0, setpoint = set_point)
        self.last_err_rate = 0

    def forward(self, input):
        self.theta.data[:] = self.relu(self.theta)
        if self.set_point<0:
            return input
        else:
            return bipolar_grad(input, self.theta, self.err_count, self.t_count)

    def init(self):
        self.last_err_rate = self.err_rate.data
        self.err_count.data[:]*=0
        self.t_count.data *= 0
        
    @property
    def err_rate(self):
        if self.t_count==0: return torch.zeros_like(self.err_count)
        return np.maximum(self.err_count/self.t_count,1e-32)
        
    def update_theta(self):
        c = self.pid(self.err_rate).to(self.theta.device)
        self.theta -= c
        self.theta.data[self.theta.data<1e-16] = 1e-16 
        self.init()
    
    def __repr__(self):
        return 'ErrorLayerBipolar err_rate {0:1.3} theta {1:1.3}'.format(self.last_err_rate.mean(), self.theta.mean())

class LenetDECOLLEErrorTriggered(LenetDECOLLEFA):
    def __init__(self, init_theta, set_point_err, lif_layer_type=LIFLayer, sign_concordant_fa = True, *args, **kwargs):
        super(LenetDECOLLEErrorTriggered, self).__init__(lif_layer_type=lif_layer_type, *args, **kwargs)

        self.err_enc_layers = nn.ModuleList()
        for i in range(len(self)):
            err_enc_layer = ErrorLayerBipolar(shape=[1],init_theta=init_theta[i], set_point = set_point_err)
            self.err_enc_layers.append(err_enc_layer)
        
    def update_thetas(self):
        for p in self.err_enc_layers:
            p.update_theta()
            
    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do, eo in zip(self.LIF_layers, 
                                         self.pool_layers, 
                                         self.readout_layers, 
                                         self.dropout_layers, 
                                         self.err_enc_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            s, u = lif(input)
            u_p = pool(u)
            s_ = smooth_step(u_p)
            sde_ = do(eo(s_))
            r = ro(sde_.reshape(sde_.size(0), -1))
            s_out.append(s_) 
            r_out.append(r)
            u_out.append(u_p)
            input = s_.detach() if lif.do_detach else s_
            i+=1

        return s_out, r_out, u_out

