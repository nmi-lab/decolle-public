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
from collections import namedtuple, OrderedDict
import warnings
from decolle.utils import train, test, accuracy, load_model_from_checkpoint, save_checkpoint, write_stats, get_output_shape, state_detach

dtype = torch.float32

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        return  (input_>0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input / (10 * torch.abs(input_) + 1.0) ** 2

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

sigmoid = nn.Sigmoid()
relu = nn.ReLU()
smooth_step = SmoothStep().apply
smooth_sigmoid = SigmoidStep().apply
fast_sigmoid = FastSigmoid.apply




class BaseLIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])
    sg_function = fast_sigmoid

    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000, do_detach=True, gain=1):
        '''
        deltat: timestep in microseconds (not milliseconds!)
        '''
        super(BaseLIFLayer, self).__init__()
        self.base_layer = layer
        self.deltat = deltat
        #self.dt = deltat/1e-6
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.beta = torch.tensor(beta, requires_grad=False)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None
        self.do_detach = do_detach
        self.gain = gain

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
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            n = layer.in_channels
            for k in layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if layer.bias is not None: 
                layer.bias.data.uniform_(-stdv, stdv)
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

    def init_parameters(self):
        self.reset_parameters(self.base_layer)

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + self.tau_s * self.gain*Sin_t #Wrong dynamics, kept for backward compatibility
        P = self.alpha * state.P + self.tau_m * state.Q #Wrong dynamics, kept for backward compatibility  
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

class LIFLayer(BaseLIFLayer):
    sg_function  = FastSigmoid.apply

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t*self.gain
        P = self.alpha * state.P + (1-self.alpha)*state.Q  
        R = self.alpharp * state.R - (1-self.alpharp)*state.S * self.wrp
        U = self.base_layer(P) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def init_parameters(self, *args, **kwargs):
        self.reset_parameters(self.base_layer, *args, **kwargs)
    
    def reset_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'):
            layer.weight.data[:] *= 1
            if layer.bias is not None:
                layer.bias.data = layer.bias.data*((1-self.alpha)*(1-self.beta))
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:] *= 5e-2
            if layer.bias is not None:
                layer.bias.data[:] = layer.bias.data[:]*((1-self.alpha)*(1-self.beta))
        else:
            warnings.warn('Unhandled data type, not resetting parameters')
            
class LIFLayerRefractory(LIFLayer):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S','U'])
    sg_function  = FastSigmoid.apply

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + (1-self.beta)*Sin_t*self.gain
        P = self.alpha * state.P + (1-self.alpha)*state.Q  
        R = self.alpharp * state.R - state.S * state.U
        U_ = self.base_layer(P)
        U = U_ + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S, U=U_)
        if self.do_detach: 
            state_detach(self.state)
        return S, U

    def init_state(self, Sin_t):
        dtype = Sin_t.dtype
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      U=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))
 #     #      
class LIFLayerNonorm(LIFLayer):
    sg_function  = smooth_step

    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + Sin_t*self.gain
        P = self.alpha * state.P + state.Q  
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + R
        S = self.sg_function(U)
        self.state = self.NeuronState(P=P, Q=Q, R=R, S=S)
        if self.do_detach: 
            state_detach(self.state)
        return S, U
    
    def reset_parameters(self, layer):
        layer.reset_parameters()
        if hasattr(layer, 'out_channels'): #its a convolution
            n = layer.in_channels
            for k in layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if layer.bias is not None:
                layer.bias.data.uniform_(-stdv, stdv)
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
        
    def randomize_tau(self, im_size, tau, std__mean = .25, tau_min = 5., tau_max = 200.):
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
        tau_v[tau_v<tau_min]=tau_min
        tau_v[tau_v>=tau_max]=tau_max
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
    output_statenames = OrderedDict(zip(['s', 'r', 'u'],[0, 1, 2]))
    def __init__(self):

        self.burnin = 0
        super(DECOLLEBase, self).__init__()

        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()

    def __len__(self):
        return len(self.LIF_layers)

    def step(self, data_batch):
        raise NotImplemented('')

    def forward(self, data_batch, doinit=True, return_sequence=False, readout_state = 'u', *args, **kwargs):
        '''
        Run network on *data_batch* sequence.
        *args*
        data_batch : Sequence has shape [batch_size, time]+[input_shape]
        doinit : Do an state init prior to running
        return_sequence : Return u of all layers and states
        '''
        if doinit: 
            state_ = self.init(data_batch)
        t_sample = data_batch.shape[1]

        if return_sequence: 
            out_ = self.step(data_batch[:,0], *args, **kwargs)
            out = [None for i in range(len(self))]
            for i in range(len(self)):
                out_i = out_[self.output_statenames[readout_state]][i]
                out[i] = torch.empty((t_sample-self.burnin,)+out_i.shape, dtype=out_i.dtype) 

        tidx = 0
        for t in (range(self.burnin,t_sample)):
            data_batch_t = data_batch[:,t]
            out_ = self.step(data_batch_t, *args, **kwargs)
            
            if return_sequence: 
                for i in range(len(self)):
                    out[i][tidx,:] = out_[self.output_statenames[readout_state]][i]
            tidx += 1

        if not return_sequence:
            ret = out_[self.output_statenames[readout_state]][-1]
        else:                                                    
            ret = out_[self.output_statenames[readout_state]][-1], out

            
        return ret 

    def name_param(self):
        return self.named_parameters()

    def get_trainable_parameters(self, layer=None):
        if layer is None:
            #might require a requires_grad check (TODO) 
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
            if not hasattr(layer, '__len__'):
                layers = [layer]
            else:
                layers = layer
            params = dict()
            for l in layers:
                cl = self.LIF_layers[l]
                for k,p in cl.named_parameters():
                    if p.requires_grad:
                        params[k]=p
            return params

    def init(self, data_batch, burnin = None):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if burnin is None:
            burnin = self.burnin
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None
        with torch.no_grad():
            for t in (range(0,max(self.burnin,1))):
                data_batch_t = data_batch[:,t]
                out_ = self.step(data_batch_t)

        for l in self.LIF_layers: state_detach(l.state)

        return [l.state for l in self.LIF_layers]

    def init_parameters(self, data_batch):
        '''
        Initialize the state and parameters
        '''
        with torch.no_grad():
            #Sin_t = data_batch[:, 0, :, :]
            #s_out, r_out = self.step(Sin_t)[:2]
            #ins = [self.LIF_layers[0].state.Q]+s_out
            for i,l in enumerate(self.LIF_layers):
                l.init_parameters()

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

    def process_output(net, data_batch):
        '''
        Process the outputs of step run over entire sequence data_batch as a continugous array.
        *data_batch*: batch of inputs, same shape as for data_batch in step()
        '''
        with torch.no_grad():
            from decolle.utils import tonp
            net.init(data_batch)
            t = (data_batch.shape[1],)
            out_states = net.step(data_batch[:,0])
            readouts = [None for _ in net.output_statenames]
            for k,v in net.output_statenames.items():
                readouts[v] = [np.zeros(t+tonp(layer).shape     ) for layer in out_states[v] if layer is not None]

            for t in range(data_batch.shape[1]):
                net.state = None
                out_states = net.step(data_batch[:,t])
                for i in range(len(net.LIF_layers)):
                    for k,v in net.output_statenames.items():
                        if out_states[v] is not None:
                            if len(out_states[v])>0:
                                if out_states[v][i] is not None:                                     
                                    readouts[v][i][t,:] = [tonp(output) for output in out_states[v][i]]

        return readouts



class DECOLLELoss(object):
    def __init__(self, loss_fn, net, reg_l = None):
        self.loss_fn = loss_fn
        self.nlayers = len(net)
        self.num_losses = len([l for l in loss_fn if l is not None])
        self.loss_layer = [i for i,l in enumerate(loss_fn) if l is not None]
        if len(loss_fn)!=self.nlayers:
            warnings.warn("Mismatch is in number of loss functions and layers. You need to specify one loss function per layer")
        self.reg_l = reg_l
        if self.reg_l is None: 
            self.reg_l = [0 for _ in range(self.nlayers)]

    def __len__(self):
        return self.nlayers

    def __call__(self, s, r, u, target, mask=1, sum_=True):
        loss_tv = []
        for i,loss_layer in enumerate(self.loss_fn):
            if loss_layer is not None:
                loss_tv.append(loss_layer(r[i]*mask, target*mask))
                if self.reg_l[i]>0:
                    uflat = u[i].reshape(u[i].shape[0],-1)
                    reg1_loss = self.reg_l[i]*((relu(uflat+.01)*mask)).mean()
                    reg2_loss = self.reg_l[i]*3e-3*relu((mask*(.1-sigmoid(uflat))).mean())
                    loss_tv[-1] += reg1_loss + reg2_loss

        if sum_:
            return sum(loss_tv)
        else:
            return loss_tv


