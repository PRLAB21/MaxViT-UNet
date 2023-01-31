import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import BACKBONES

# Segmentation model with 
@BACKBONES.register_module()
class EnsembleSegNet(BaseModule):
    def init__(self, model1, model2, model3, input_ch = 3):
        super(EnsembleSegNet, self).__init__()
        
        self.m1 = model1
        self.m2 = model2
        
