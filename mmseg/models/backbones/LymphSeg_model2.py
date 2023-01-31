#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:50:08 2022

@author: zunaira
"""

# import math
# from collections import OrderedDict

# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F

from .LmphSegNet_utils import *
# from .LymphSegNet_utils import (Net, MSST, RDT, RAT, encoder_downsample, decoder_upsample)
# from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer, UpSample2x)
# from .utils import crop_op, crop_to_shape
from mmcv.runner import BaseModule
from ..builder import BACKBONES

####
@BACKBONES.register_module()
class SegLymphNet2(BaseModule): 
    """Initialise SegLymph-Net."""

    def __init__(self, input_ch=3, debug=False):
        super(SegLymphNet2, self).__init__()
        self.module_name = '[SegLymphNet2]'
        self.debug = debug
        self.output_ch = 1 


        def bottle_neck(in_ch=256, out_ch=512, ksize=3):
            b1 = RAT(in_ch, out_ch)
            return b1


        self.e1 = nn.Sequential(RDT(input_ch, 32),
                                MSST(32, 32),
                                RDT(32, 32))
        self.pool1 = encoder_downsample('maxpool', 2, 2)
        self.e2 = nn.Sequential(RDT(32, 64),
                                MSST(64, 64),
                                RDT(64, 64))
                                # RDT(128, 128),
        self.pool2 = encoder_downsample('maxpool', 2, 2)
        self.e3 = nn.Sequential(RDT(64, 128),
                                MSST(128, 128),
                                RDT(128, 128))
        self.pool3 = encoder_downsample('maxpool', 2, 2)
        self.e4 = nn.Sequential(RDT(128, 256),
                                MSST(256, 256),
                                RDT(256, 256))
        self.pool4 = encoder_downsample('maxpool', 2, 2)
            # encoder = nn.Sequential(OrderedDict([("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)]))
            # return encoder


        self.bottle_neck = bottle_neck()


        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.d4 = nn.Sequential(nn.Conv2d(512, 256, 3, stride=1, padding=1, bias=False),
                                nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
                                nn.ReLU(inplace=True),
                                RAT(256, 256))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = nn.Sequential(nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=False),
                                nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
                                nn.ReLU(inplace=True),
                                RAT(128, 128))
                #(nn.Conv2d(128, 128, 1, stride=1, padding=0, bias=False),
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = nn.Sequential(nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
                                nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
                                nn.ReLU(inplace=True),
                                RAT(64, 64))
            
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
                                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                                nn.ReLU(inplace=True),
                                RAT(32, 32))
        self.last = nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=False)



        #  self.upsample2x = decoder_upsample(2)
        # TODO: pytorch still require the channel eventhough its ignored
        # self.weights_init()

    def forward(self, x):
        TAG = self.module_name + '[forward]'
        outputs = []
        if self.debug: print(TAG, '[x]', x.shape)

        # Encoder
        enc1 = self.e1(x)
        if self.debug: print(TAG, '[enc1]', enc1.shape)
        enc2 = self.e2(self.pool1(enc1))
        if self.debug: print(TAG, '[enc2]', enc2.shape)
        enc3 = self.e3(self.pool2(enc2))
        if self.debug: print(TAG, '[enc3]', enc3.shape)
        enc4 = self.e4(self.pool3(enc3))
        if self.debug: print(TAG, '[enc4]', enc4.shape)

        # Bottleneck
        bottleneck = self.bottle_neck(self.pool4(enc4))
        if self.debug: print(TAG, '[bottleneck]', bottleneck.shape)
        outputs.append(bottleneck)

        # Decoder
        dec4 = self.up4(bottleneck)
        if self.debug: print(TAG, '[dec4][up4]', dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)
        if self.debug: print(TAG, '[dec4][cat]', dec4.shape)
        dec4 = self.d4(dec4)
        if self.debug: print(TAG, '[dec4][d4]', dec4.shape)
        outputs.append(dec4)

        dec3 = self.up3(dec4)
        if self.debug: print(TAG, '[dec3][up3]', dec3.shape)
        dec3 = torch.cat((dec3, enc3), dim=1)
        if self.debug: print(TAG, '[dec3][cat]', dec3.shape)
        dec3 = self.d3(dec3)
        if self.debug: print(TAG, '[dec3]', dec3.shape)
        outputs.append(dec3)

        dec2 = self.up2(dec3)
        if self.debug: print(TAG, '[dec2][up2]', dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        if self.debug: print(TAG, '[dec2][cat]', dec2.shape)
        dec2 = self.d2(dec2)
        if self.debug: print(TAG, '[dec2]', dec2.shape)
        outputs.append(dec2)

        dec1 = self.up1(dec2)
        if self.debug: print(TAG, '[dec1][up1]', dec1.shape)
        dec1 = torch.cat((dec1, enc1), dim=1)
        if self.debug: print(TAG, '[dec1][cat]', dec1.shape)
        dec1 = self.d1(dec1)
        if self.debug: print(TAG, '[dec1]', dec1.shape)
        outputs.append(dec1)

        return outputs
        # return torch.sigmoid(self.last(dec1))
