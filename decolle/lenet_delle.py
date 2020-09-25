#!/bin/python
#-----------------------------------------------------------------------------
# File Name : allconv_decolle.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from decolle.base_model import *
import torch.utils.data
import torchvision

def one_hot(y, out_channels = 10):
    y_onehot = torch.FloatTensor(y.shape[0], out_channels)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot

def one_hot_np(y, out_channels = 10):
    y_onehot = torch.FloatTensor(1,out_channels)
    y_onehot.zero_()
    y_onehot[0,y] = 1
    # In your for loop
    return y_onehot

class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

class LenetDELLE(DECOLLEBase):
    requires_init = False
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lc_ampl=1.5,
                 method = 'rtrl'):
        self.local = True if method is 'rtrl' else False
        self.lc_ampl = lc_ampl
        num_layers = num_conv_layers + num_mlp_layers

        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride=[1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if Nhid is None:          self.Nhid = Nhid = []
        if Mhid is None:          self.Mhid = Mhid = []


        super(LenetDELLE, self).__init__()

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

        for i in range(num_conv_layers):
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]

            layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])

            pool = nn.MaxPool2d(kernel_size=pool_size[i])

            readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

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
            layer = nn.Linear(Mhid[i], Mhid[i+1])
            readout = nn.Linear(Mhid[i+1], out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[self.num_conv_layers+i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            u = lif(input)
            u_p = pool(u)
            u_sig = do(torch.sigmoid(u_p))
            r = ro(u_sig.reshape(u_sig.size(0), -1))
            #non-linearity must be applied after the pooling to ensure that the same winner is selected
            s_out.append(torch.sigmoid(u_p)) 
            r_out.append(r)
            u_out.append(u_p)
            input = torch.sigmoid(u_p.detach() if self.local else u_p)
            i+=1

        return s_out, r_out, u_out

def create_data(batch_size_train=32, batch_size_test=32):
    train_loader = torch.utils.data.DataLoader(
                           torchvision.datasets.MNIST('datamnist/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                               torchvision.transforms.Lambda(lambda x: x.unsqueeze(1))
                             ]),
                             target_transform = torchvision.transforms.Compose([
                               torchvision.transforms.Lambda(lambda x: one_hot_np(x))
                             ])
                             ),
                           batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                           torchvision.datasets.MNIST('datamnist/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                               torchvision.transforms.Lambda(lambda x: x.unsqueeze(1))
                             ]),
                             target_transform = torchvision.transforms.Compose([
                                 torchvision.transforms.Lambda(lambda x: one_hot_np(x))
                             ])
                             ),
                           batch_size=batch_size_test, shuffle=False)
    return train_loader, test_loader

class GlobalLoss(DECOLLELoss):
    def __len__(self):
        return 1

    def __call__(self, s, r, u, target, mask=1, sum_=True):
        loss_tv = []
        i = self.nlayers-1
        uflat = u[i].reshape(u[i].shape[0],-1)
        loss_tv.append(self.loss_fn(r[i]*mask, target*mask))
        if self.reg_l[i]>0:
            reg1_loss = self.reg_l[i]*1e-2*((relu(uflat+.01)*mask)).mean()
            reg2_loss = self.reg_l[i]*6e-5*relu((mask*(.1-sigmoid(uflat))).mean())
            loss_tv[-1] += reg1_loss + reg2_loss

        if sum_:
            return sum(loss_tv)
        else:
            return loss_tv
