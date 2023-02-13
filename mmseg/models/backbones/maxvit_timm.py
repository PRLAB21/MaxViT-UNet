# import torch
import torch.nn as nn
# from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
# from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
#                                         trunc_normal_)
# from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
#                          load_state_dict)
# from torch.nn.modules.batchnorm import _BatchNorm
# from torch.nn.modules.utils import _pair as to_2tuple

from ..builder import BACKBONES

import timm


@BACKBONES.register_module()
class MaxVitTimm(nn.Module):

    supported_maxvit_archs = timm.list_models("*maxvit*")

    def __init__(self, arch, pretrained=True, out_classes=10):
        super(MaxVitTimm, self).__init__()

        assert arch in self.supported_maxvit_archs, f'Argument arch must be one of {self.supported_maxvit_archs}, but found {arch}'

        maxvit_model = timm.create_model(arch, pretrained=True, num_classes=out_classes)
        self.stem = maxvit_model.stem
        self.stages = maxvit_model.stages
        self.norm = maxvit_model.norm
        self.head = maxvit_model.head

        # self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=self.maxvit_nano.num_features, out_features=feature_space),
        #     nn.SiLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=feature_space, out_features=out_classes)
        # )

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            outs.append(x)
        x = self.norm(x)
        x = self.head(x)
        return outs
