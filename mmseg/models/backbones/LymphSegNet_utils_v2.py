#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:12:55 2022

@author: zunaira
"""
import torch
# from collections import OrderedDict
# import torch.nn.functional as F
# import torchvision.models as tmodels
from torch.nn import ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, \
    init, AvgPool2d, LeakyReLU, AdaptiveMaxPool2d, Sigmoid, AdaptiveAvgPool2d
#from torchsummary import summary
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import torch.optim as optim
# import cv2

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

def conv_1x1_bnrelu(in_chnl, out_chnl):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(out_chnl),
            LeakyReLU(negative_slope=0.02, inplace=True),)
            
def conv_1x1_bn(in_chnl, out_chnl):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=1, stride=1, padding=0),
            BatchNorm2d(out_chnl))
            # LeakyReLU(negative_slope=0.02, inplace=True),)

def conv_3x3_bnrelu(in_chnl):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=in_chnl, kernel_size=3, dilation= 1, stride=1, padding= 1),
            BatchNorm2d(in_chnl),
            LeakyReLU(negative_slope=0.02, inplace=True),)

def conv_3x3_bn(in_chnl, out_chnl, dil):
    return Sequential(Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=3, dilation= dil, stride=1, padding= dil),
            BatchNorm2d(out_chnl))
           
def conv2_block(in_ch):
    return Sequential( conv_3x3_bn(in_ch, in_ch, 1),
            conv_3x3_bn(in_ch, in_ch, 1),
            LeakyReLU(negative_slope=0.02, inplace=True)
            )
    
def conv3_block(in_ch):
   return Sequential(
            conv_3x3_bn(in_ch, in_ch, 1),
            conv_3x3_bn(in_ch, in_ch, 1),
            conv_3x3_bn(in_ch, in_ch, 1),
            LeakyReLU(negative_slope= 0.02, inplace=True))

def avg_pool(k, s):
    return AvgPool2d(kernel_size=k, stride=s)

def max_pool (k,s):
    return MaxPool2d(kernel_size=k, stride=s)

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
    def __init__(self, in_channels):
        super(MSST, self).__init__()
        self.in_channels= in_channels
       
        self.init_convs = Sequential(conv3_block(in_channels), conv_3x3_bnrelu(in_channels))

        self.convA = Sequential(conv2_block(in_channels),  
                                conv_3x3_bn(in_channels, in_channels, 1), 
                                conv_3x3_bnrelu(in_channels)
                                )

        self.convB = Sequential(conv2_block(in_channels),  
                                conv_3x3_bn(in_channels, in_channels, 2), 
                                conv_3x3_bnrelu(in_channels)
                                )
        self.convC = Sequential(conv2_block(in_channels),  
                                conv_3x3_bn(in_channels, in_channels, 4), 
                                conv_3x3_bnrelu(in_channels)
                                )

        # self.convB1 = Sequential(conv2_block(in_channels), 
        #                         conv_3x3_bn(in_channels, in_channels, 2), conv_3x3_bnrelu(in_channels))
        # self.convB2 = Sequential(conv_3x3_bn(in_channels, in_channels, 1), 
        #                         conv_3x3_bn(in_channels, in_channels, 2), conv_3x3_bnrelu(in_channels))
        # self.convB3 = Sequential(conv2_block(in_channels), 
        #                         conv_3x3_bn(in_channels, in_channels, 2), conv_3x3_bnrelu(in_channels))

        # self.convC1 = Sequential(conv2_block(in_channels), 
        #                         conv_3x3_bn(in_channels, in_channels, 4), conv_3x3_bnrelu(in_channels))
        # self.convC2 = Sequential(conv_3x3_bn(in_channels, in_channels, 1), 
        #                         conv_3x3_bn(in_channels, in_channels, 4), conv_3x3_bnrelu(in_channels))
        # self.convC3 = Sequential(conv2_block(in_channels), 
        #                         conv_3x3_bn(in_channels, in_channels, 4), conv_3x3_bnrelu(in_channels))

        self.end_convs = Sequential(conv3_block(in_channels))


    def forward(self, x):
        y = self.init_convs(x)
        # Branch A
        yA = self.convA(y)
        # Branch B
        yB = self.convB(y)
        # Branch C
        yC = self.convC(y)
        # adding the three braches
        ycomb = yA + yB +yC
        # going to 3rd stage of convs
        y2 = self.end_convs(ycomb)
        return y2
    
# Residual Transform Block
class RET(Module):
    def __init__(self, in_ch):
        super(RET, self).__init__()
        self.conv_block = Sequential(
                                conv2_block(in_ch), 
                                # conv3_block(in_ch),
                                conv_3x3_bn(in_ch, in_ch, 1)
                                )
        # self.conv2 = Sequential(
        #                         conv2_block(in_ch), 
        #                         # conv2_block(in_ch),
        #                         conv_3x3_bn(in_ch, in_ch, 1)
        #                         )
        # self.conv3 = Sequential(
        #                         conv2_block(in_ch),  
        #                         conv_3x3_bn(in_ch, in_ch, 1)
        #                         # conv_3x3(in_ch, out_ch, 2)
        #                         )
        # self.conv4 = Sequential(
        #                         # conv_3x3_bn(in_ch, in_ch, 1), 
        #                         conv_3x3_bn(in_ch, in_ch, 1)
        #                         # conv_3x3(in_ch, out_ch, 4)
        #                         )
        # self.downsample = conv_1x1_bn(in_ch, in_ch)
        self.relu = LeakyReLU(negative_slope=0.02, inplace=True)

    def forward(self, x):
        y1 = self.conv_block(x)
        y2 = self.conv_block(y1)

        s2 = self.relu(x + y2)                        
        return s2

        # y3 = self.conv_block(s2)
        # y4 = self.conv_block(y3)
        #s3 = self.relu(s2 + y4)

        #return s3


        # y_out = y + self.downsample(x)
        


# Attention Transform block
class ATT(Module):
    def __init__(self, in_ch):
        super(ATT, self).__init__()
        self.conv = Sequential(
                               conv_1x1_bnrelu(in_ch, in_ch),
                               conv_3x3_bnrelu(in_ch),
                               conv_1x1_bnrelu(in_ch, in_ch)
                               )
        self.ca = ChannelAttention(in_ch)
        self.sa = SpatialAttention()
        self.downsample = conv_1x1_bn(in_ch, in_ch)

    def forward(self, x):
        y = self.conv(x)
        # print('[RAT][y][conv]', y.shape)
        y = y * self.ca(y)
        # print('[RAT][y][ca]', y.shape)
        y = y * self.sa(y)
        # print('[RAT][y][sa]', y.shape)
        y_out = y + self.downsample(x)
        # print('[RAT][y_out]', y_out.shape)
        return y_out

# Region uniformity-based CNN
class REU(Module):
    def __init__(self, in_ch):
        super(REU, self).__init__()
        self.conv1 = conv3_block(in_ch)
        self.pool = avg_pool(2)
        self.conv2 = conv3_block(in_ch)
        self.conv3 = conv2_block(in_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool(y)
        y = self.conv2(x)
        y = self.pool(y)
        y = self.conv3(x)
        return y

