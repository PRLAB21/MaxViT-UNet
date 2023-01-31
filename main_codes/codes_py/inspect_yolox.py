# import os
# import cv2
# import glob
# import time
# import json
# import numpy as np
# import pandas as pd
# import ml_metrics as metrics
# import matplotlib.pyplot as plt
# from tqdm import tqdm
from pprint import pprint

import torch
# import torch.nn.functional as F
# from torch.nn import Module, Conv2d, BatchNorm2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, ReLU, Sequential, Linear, Dropout, Softmax

# import mmcv
from mmcv import Config
# from mmcv.cnn import fuse_conv_bn
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model, build_runner)
# from mmdet.utils import get_root_logger

from mmdet.datasets import build_dataloader, build_dataset
# from mmdet.models import build_detector, BACKBONES, ResNet, ResNeXt, TridentResNet, MobileNetV2, PyramidVisionTransformerV2, RegNet, Res2Net, ResNeSt, SwinTransformer
from mmdet.apis import set_random_seed #, train_detector, inference_detector, init_detector, show_result_pyplot, multi_gpu_test, single_gpu_test

set_random_seed(0, deterministic=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')


## Learning about Dataset
def inspect_dataset():
    global train_data
    TAG2 = TAG + '[inspect_dataset]'

    train_data = build_dataset(cfg.data.train)
    print(TAG2, '[train_data]', type(train_data), type(train_data.data_infos))
    print(TAG2, '[train_data.pipeline]\n', train_data.pipeline)

    idx = 1
    img_info = train_data.data_infos[idx]
    print(TAG2, '[img_info]\n', img_info)

    ann_info = train_data.get_ann_info(idx)
    print(TAG2, '[ann_info]\n', ann_info)

    results = dict(img_info=img_info, ann_info=ann_info)
    print(TAG2, '[results]')
    pprint(results, depth=4)

    # train_data.pre_pipeline(results)
    # print(TAG2, '[results]')
    # pprint(results, depth=4)

    # train_data.pipeline(results)
    # print(TAG2, '[results]')
    # pprint(results)

    x = train_data[1]
    print(TAG2, '[x]')
    pprint(x)
    print(TAG2, '[x[img]]')
    print(type(x['img']), dir(x['img']))


## Learning about DataLoader
def inspect_dataloader():
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        runner_type=cfg.runner['type'],
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    print('[train_loader_cfg]')
    pprint(train_loader_cfg)

    train_loader = build_dataloader(train_data, **train_loader_cfg)
    train_loader

    dir(train_loader)

    next(iter(train_loader))


## Learning about Model
def inspect_model():
    model = build_detector(cfg.model)
    model

    x = torch.rand((1, 3, 224, 224))
    y_backbone = model.backbone(x)
    print('[y_backbone]', len(y_backbone))
    for level, y_level in enumerate(y_backbone):
        print(f'[y_backbone][level{level}]', y_level.shape)

    y_neck = model.neck(y_backbone)
    print('[y_neck]', len(y_neck))
    for level, y_level in enumerate(y_neck):
        print(f'[y_neck][level{level}]', y_level.shape)

    rpn_train_cfg = cfg.model.train_cfg.rpn
    rpn_train_cfg

    rpn_head_ = cfg.model.rpn_head.copy()
    rpn_head_

    rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=cfg.model.test_cfg.rpn)
    rpn_head_

    # model.rpn_head.prior_generator.get_anchors()
    rpn_cls_scores, rpn_bbox_preds = model.rpn_head(y_neck)
    print('[rpn_cls_scores, rpn_bbox_preds]', len(rpn_cls_scores), len(rpn_bbox_preds))
    for level, (rpn_cls_score, rpn_bbox_pred) in enumerate(zip(rpn_cls_scores, rpn_bbox_preds)):
        print(f'[rpn_cls_scores, rpn_bbox_preds][level{level}]', rpn_cls_score.shape, rpn_bbox_pred.shape)


TAG = '[inspect_yolox]'
config_file_path = 'configs/lyon/yolox_s_s1_lyon.py'
cfg = Config.fromfile(config_file_path)
print(f'Config:\n{cfg.pretty_text}')

inspect_dataset()
