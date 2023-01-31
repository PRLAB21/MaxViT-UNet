#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:55 2022

@author: zunaira
"""
import torch
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as tmodels
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout, init, AvgPool2d, LeakyReLU, Dropout2d, ModuleDict, ZeroPad2d, ReflectionPad2d, AdaptiveMaxPool2d, Sigmoid, AdaptiveAvgPool2d
#from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.optim as optim
import cv2

# creating a new segmentation model
 
 
"""""
architecture contains three different types of blocks, 1) MSSD, 2) RDT and 3) RAT in its encoder and decoder design
"""""

class Net(Module):
    """ A base class provides a common weight initialisation scheme."""

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return 
 ##################################################
 ##################################################
 
def conv_1x1(in_chnl, out_chnl):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(out_chnl),
            LeakyReLU(negative_slope=0.02, inplace=True),)
def conv_3x3(in_chnl, out_chnl, dil):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=3, dilation= dil, stride=1, padding= dil),
            BatchNorm2d(out_chnl),
            LeakyReLU(negative_slope=0.02, inplace=True),)

def avg_pool(k, s):
    return AvgPool2d(kernel_size= k, stride= s)

def max_pool (k,s):
    return MaxPool2d(kernel_size= k, stride= s)

def encoder_downsample (pool, k, s):
    if pool == 'avgpool':
        return avg_pool(k, s)
    else:
        return max_pool(k, s)
    
def decoder_upsample (in_ch, out_ch, k, s, p):
    return Sequential(
            Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding= p),
            BatchNorm2d(out_ch),
            LeakyReLU(negative_slope=0.02, inplace=True))

class ChannelAttention(Module):
        def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = AdaptiveAvgPool2d(1)
            self.max_pool = AdaptiveMaxPool2d(1)
   
            self.fc = Sequential(Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                  ReLU(),
                                   Conv2d(in_planes // 16, in_planes, 1, bias=False))
            self.sigmoid = Sigmoid()
    
        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            return self.sigmoid(out)
    
class SpatialAttention(Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()
            self.conv1 = Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
            self.sigmoid = Sigmoid()
        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv1(x)
            return self.sigmoid(x)
   

# multi-scale split transform block MSST
class MSST(Module):
    def __init__(self, in_channels, out_channels):
        super(MSST, self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_1 = conv_1x1(in_channels, out_channels)
        self.conv3 = conv_3x3(out_channels, out_channels, dil = 1)
        self.conv3D = conv_3x3(out_channels, out_channels, dil = 2)
        # self.conv2_3 = conv_3x3(out_channels, out_channels, dil = 1)
        # self.conv2_4 = conv_3x3(out_channels, out_channels, dil = 2)
        # self.conv3_1 = conv_3x3(out_channels, out_channels, dil = 1)
        # self.conv3_2 = conv_3x3(out_channels, out_channels, dil = 1)
     
    
    def forward(self, x):
        # x has 3 channels
        y = self.conv_1(x)
        # x_out has 32 channels
        y2_1 = self.conv3(y)
        y2_2 = self.conv3D(y)
        y2_3 = self.conv3(y)
        y2_4 = self.conv3D(y)
        # adding 1st two convs
        y3_1 = y2_1 + y2_2
        # adding 2nd two convs
        y3_2 = y2_3 + y2_4
        # going to 3rd stage of convs
        y4_1 = self.conv3(y3_1)
        y4_2 = self.conv3(y3_2)
        y5 = y4_1 + y4_2
        y5_1 = self.conv3(y5)
        y5_2 = self.conv3(y5_1)
        y6 = self.conv3(y + y5_2)
        return y6
    
# Residual Dilated Transform Block
class RDT(Module):
    def __init__(self, in_ch, out_ch):
        super(RDT, self).__init__()
        self.conv1 = Sequential(
                                conv_3x3(in_ch, out_ch, 1), 
                                conv_3x3(out_ch, out_ch, 1),
                                conv_3x3(out_ch, out_ch, 1)
                                )
        self.conv2 = Sequential(
                                conv_3x3(out_ch, out_ch, 1), 
                                conv_3x3(out_ch, out_ch, 1),
                                conv_3x3(out_ch, out_ch, 1)
                                )
        self.conv3 = Sequential(
                                conv_3x3(out_ch, out_ch, 2), 
                                conv_3x3(out_ch, out_ch, 2),
                                # conv_3x3(in_ch, out_ch, 2)
                                )
        self.conv4 = Sequential(
                                conv_3x3(out_ch, out_ch, 4), 
                                conv_3x3(out_ch, out_ch, 4),
                                # conv_3x3(in_ch, out_ch, 4)
                                )
        self.skip = conv_1x1(in_ch, out_ch)
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        x_1 = self.skip(x)
        y_out = y+x_1
        return y_out
                          

# Residua Attention Transform
class RAT(Module):
    def __init__(self, in_ch, out_ch):
        super(RAT, self).__init__()
        self.conv = Sequential(conv_1x1(in_ch, out_ch),
                                conv_3x3(out_ch, out_ch, 1),
                                conv_1x1(out_ch, out_ch))
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()
        self.skip = conv_1x1(in_ch, out_ch)
    def forward(self, x):
        y = self.conv(x)
        y = y*self.ca(y)
        y = y*self.sa(y)
        y_out = y+self.skip(x)
        return y_out