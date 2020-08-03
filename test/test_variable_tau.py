#!/bin/python
#-----------------------------------------------------------------------------
# File Name : test_variable_tau.py
# Author: Emre Neftci
#
# Creation Date : Sun 15 Sep 2019 11:06:14 PM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from decolle.base_model import *

input_shape = [32, 100, 100]
c = torch.nn.Conv2d(32,64,5)
layer = LIFLayerVariableTau(c)

input_ = torch.zeros([50]+input_shape)
layer.init_parameters(input_)
layer(input_)

from decolle.lenet_decolle_model_errortriggered import LenetDECOLLEErrorTriggered
net = LenetDECOLLEErrorTriggered(input_shape=input_shape).cuda()
out = net(input_.cuda())