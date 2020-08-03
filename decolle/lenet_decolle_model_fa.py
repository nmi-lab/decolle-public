#!/bin/python
#-----------------------------------------------------------------------------
# File Name : lenet_decolle_model_pa.py
# Author: Emre Neftci
#
# Creation Date : Tue 04 Feb 2020 11:51:17 AM PST
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 

from .base_model import *
from .lenet_decolle_model import LenetDECOLLE
import math

class LenetDECOLLEFA(LenetDECOLLE):
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 deltat=1000,
                 lc_ampl=.5,
                 lif_layer_type = LIFLayer,
                 method='rtrl',
                 with_output_layer=False,
                 sign_concordant_fa = True):
        self.sign_concordant_fa = sign_concordant_fa
        self.with_output_layer = with_output_layer
        if with_output_layer:
            Mhid += [out_channels]
            num_mlp_layers += 1
        num_layers = num_conv_layers + num_mlp_layers
        self.num_layers = num_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride=[1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if len(alpha) == 1:         alpha = alpha * num_layers
        if len(alpharp) == 1:       alpharp = alpharp * num_layers
        if len(beta) == 1:          beta = beta * num_layers
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if Nhid is None:          self.Nhid = Nhid = []
        if Mhid is None:          self.Mhid = Mhid = []

        DECOLLEBase.__init__(self)

        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2  # TODO try to remove padding



        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]

        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = lif_layer_type(base_layer,
                             alpha=alpha[i],
                             beta=beta[i],
                             alpharp=alpharp[i],
                             deltat=deltat,
                             do_detach= True if method == 'rtrl' else False)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            readout = FALinear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        Mhid = [mlp_in] + Mhid
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i+1])
            layer = lif_layer_type(base_layer,
                             alpha=alpha[i],
                             beta=beta[i],
                             alpharp=alpharp[i],
                             deltat=deltat,
                             do_detach= True if method == 'rtrl' else False)
            
            if self.with_output_layer and i+1==num_mlp_layers:
                readout = nn.Identity()
                dropout_layer = nn.Identity()
            else:
                readout = FALinear(Mhid[i+1], out_channels)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
                self.reset_lc_parameters(readout, lc_ampl)
                dropout_layer = nn.Dropout(dropout[self.num_conv_layers+i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / math.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)
             
        if hasattr(layer, 'weight_fa') and self.sign_concordant_fa:
            #Implements sign concordance
            print('Using sign-cordant FA')
            layer.weight_fa.data.normal_(1, .5)
            layer.weight_fa.data[layer.weight_fa.data<0] = 0
            layer.weight_fa.data[:] *= layer.weight.data[:]
        elif hasattr(layer, 'weight_fa'):
            print('Using non sign-cordant FA')
            m = layer.weight.data.min()
            s = layer.weight.data.max()
            layer.weight_fa.data.uniform_(m, s)

if __name__ == "__main__":
    #Test building network
    net = LenetDECOLLEFA(Nhid=[1,8],Mhid=[32,64],out_channels=10, input_shape=[1,28,28])
    d = torch.zeros([1,1,28,28])
    net(d)


