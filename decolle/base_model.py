#!/bin/python
# -----------------------------------------------------------------------------
# File Name : multilayer.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 12-03-2019
# Last Modified : Tue 12 Mar 2019 04:51:44 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
# -----------------------------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from itertools import chain
from collections import namedtuple
import warnings
from decolle.utils import train, test, accuracy, load_model_from_checkpoint, save_checkpoint, write_stats, get_output_shape

dtype = torch.float32

sigmoid = nn.Sigmoid()
relu = nn.ReLU()

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >=0).type(x.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input
    
class SigmoidStep(torch.autograd.Function):
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >=0).type(x.dtype)

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        res = torch.sigmoid(input)
        return res*(1-res)*grad_output

smooth_step = SmoothStep().apply
smooth_sigmoid = SigmoidStep().apply

class LinearFAFunction(torch.autograd.Function):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            # all of the logic of FA resides in this one line
            # calculate the gradient of input with fixed fa tensor, rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if context.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class FALinear(nn.Module):
    '''from https://github.com/L0SG/feedback-alignment-pytorch/'''
    def __init__(self, input_features, output_features, bias=True):
        super(FALinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass
        # does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # weight initialization
        #torch.nn.init.kaiming_uniform(self.weight)
        #torch.nn.init.kaiming_uniform(self.weight_fa)
        #torch.nn.init.constant(self.bias, 1)
        # does not need gradient
        self.weight_fa = torch.nn.Parameter(torch.FloatTensor(output_features, input_features), requires_grad=False)

        # weight initialization
        #torch.nn.init.kaiming_uniform(self.weight)
        #torch.nn.init.kaiming_uniform(self.weight_fa)
        #torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)

def state_detach(state):
    for s in state:
        s.detach_()

class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    sg_function = smooth_step

    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super(LIFLayer, self).__init__()
        self.base_layer = layer
        self.deltat = deltat
        self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled layer type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_t):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + self.tau_s * Sin_t
        P = self.alpha * state.P + self.tau_m * state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if hasattr(layer, 'out_channels'):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features
    
    def get_device(self):
        return self.base_layer.weight.device
    
class LIFLayerNonorm(LIFLayer):
    sg_function  = smooth_step
    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + Sin_t
        P = self.alpha * state.P + state.Q  # TODO check with Emre: Q or state.Q?
        R = self.alpharp * state.R - state.S * self.wrp
        #Pc = (P>self.cutoff).type(P.dtype)/self.cutoff
        #Pd = (P-Pc).detach()+Pc
        U = self.base_layer(P) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def reset_parameters(self, layer):
        if hasattr(layer, 'out_channels'):
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250 * self.tau_s * self.tau_m
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warnings.warn('Unhandled data type, not resetting parameters')

class LIFLayerVariableTau(LIFLayer):
    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, random_tau=True, do_detach=True):
        super(LIFLayerVariableTau, self).__init__(layer, alpha, alpharp, wrp, beta, deltat)
        self.random_tau = random_tau
        self.alpha_mean = self.alpha
        self.beta_mean = self.beta
        self.do_detach = do_detach
        
    def randomize_tau(self, im_size, tau, std__mean = .25):
        '''
        Returns a random (normally distributed) temporal constant of size im_size computed as
        `1 / Dt*tau where Dt is the temporal window, and tau is a random value expressed in microseconds
        between low and high.
        :param im_size: input shape
        :param mean__std: mean to standard deviation
        :return: 1/Dt*tau
        '''
        tau_v = torch.empty(im_size)
        tau_v.normal_(1, std__mean)
        tau_v.data[:] *= tau 
        tau_v[tau_v<5]=5
        tau_v[tau_v>=200]=200
        #tau = np.broadcast_to(tau, (im_size[0], im_size[1], channels)).transpose(2, 0, 1)
        return torch.Tensor(1 - 1. / tau_v)    
    
    def init_parameters(self, Sin_t):
        device = self.get_device()
        input_shape = list(Sin_t.shape)
        if self.random_tau:
            tau_m = 1./(1-self.alpha_mean)
            tau_s = 1./(1-self.beta_mean)
            self.alpha = self.randomize_tau(input_shape[1:], tau_m).to(device)
            self.beta  = self.randomize_tau(input_shape[1:], tau_s).to(device)
        else:
            tau_m = 1./(1-self.alpha_mean)
            tau_s = 1./(1-self.beta_mean)
            self.alpha = torch.ones(input_shape[1:]).to(device)*self.alpha_mean.to(device)
            self.beta  = torch.ones(input_shape[1:]).to(device)*self.beta_mean.to(device)
        self.alpha = self.alpha.view(Sin_t.shape[1:])
        self.beta  = self.beta.view(Sin_t.shape[1:])
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad = False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad = False)
        self.reset_parameters(self.base_layer)

class DECOLLEBase(nn.Module):
    requires_init = True
    def __init__(self):

        super(DECOLLEBase, self).__init__()

        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()

    def __len__(self):
        return len(self.LIF_layers)

    def forward(self, input):
        raise NotImplemented('')
    
    @property
    def output_layer(self):
        return self.readout_layers[-1]

    def get_trainable_parameters(self, layer=None):
        if layer is None:
            return chain(*[l.parameters() for l in self.LIF_layers])
        else:
            return self.LIF_layers[layer].parameters()

    def get_trainable_named_parameters(self, layer=None):
        if layer is None:
            params = dict()
            for k,p in self.named_parameters():
                if p.requires_grad:
                    params[k]=p

            return params
        else:
            return self.LIF_layers[layer].named_parameters()

    def init(self, data_batch, burnin):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None
        with torch.no_grad():
            for i in range(max(len(self), burnin)):
                self.forward(data_batch[:, i, :, :])

    def init_parameters(self, data_batch):
        Sin_t = data_batch[:, 0, :, :]
        s_out, r_out = self.forward(Sin_t)[:2]
        ins = [self.LIF_layers[0].state.Q]+s_out
        for i,l in enumerate(self.LIF_layers):
            l.init_parameters(ins[i])

    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        self.reset_lc_bias_parameters(layer,lc_ampl)

    def reset_lc_bias_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)
    
    def get_input_layer_device(self):
        if hasattr(self.LIF_layers[0], 'get_device'):
            return self.LIF_layers[0].get_device() 
        else:
            return list(self.LIF_layers[0].parameters())[0].device

    def get_output_layer_device(self):
        return self.output_layer.weight.device 


class DECOLLELoss(object):
    def __init__(self, loss_fn, net, reg_l = None):
        self.loss_fn = loss_fn
        self.nlayers = len(net)
        self.num_losses = len([l for l in loss_fn if l is not None])
        assert len(loss_fn)==self.nlayers, "Mismatch is in number of loss functions and layers. You need to specify one loss function per layer"
        self.reg_l = reg_l
        if self.reg_l is None: 
            self.reg_l = [0 for _ in range(self.nlayers)]

    def __len__(self):
        return self.nlayers

    def __call__(self, s, r, u, target, mask=1, sum_=True):
        loss_tv = []
        for i,loss_layer in enumerate(self.loss_fn):
            if loss_layer is not None:
                uflat = u[i].reshape(u[i].shape[0],-1)
                loss_tv.append(loss_layer(r[i]*mask, target*mask))
                if self.reg_l[i]>0:
                    reg1_loss = self.reg_l[i]*1e-2*((relu(uflat+.01)*mask)).mean()
                    reg2_loss = self.reg_l[i]*6e-5*relu((mask*(.1-sigmoid(uflat))).mean())
                    loss_tv[-1] += reg1_loss + reg2_loss

        if sum_:
            return sum(loss_tv)
        else:
            return loss_tv


