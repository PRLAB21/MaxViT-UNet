import gc

import torch

import mmcv
from mmcv import Config

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import set_random_seed, train_detector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

set_random_seed(0, deterministic=False)

config_file_path = './configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s11_lysto.py'
cfg = Config.fromfile(config_file_path)
print(f'Config:\n{cfg.pretty_text}')

gc.collect()
torch.cuda.empty_cache()

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model)
model.cfg = cfg

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# craete working directory to store logs and model checkpoints
mmcv.mkdir_or_exist(cfg.work_dir)

train_detector(model, datasets, cfg, distributed=False, validate=True)
