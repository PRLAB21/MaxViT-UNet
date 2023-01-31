import os
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet.apis import init_detector
from mmdet.core.utils.model_output_save_hook import ModelOutputSaveHook

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

TAG = '[model_output_save_hook]'
path_config = './configs/mask_rcnn_lymphocyte/maskrcnn_lymphocytenet3_cm1_s6_lysto.py'
cfg = Config.fromfile(path_config)

epoch = 30
cfg.load_from = os.path.join(cfg.PATH_WORK_DIR, f'epoch_{epoch}.pth')
print(cfg.load_from, os.path.exists(cfg.load_from))
cfg.resume_from = ''

model = init_detector(cfg, cfg.load_from)
model.CLASSES = cfg.classes
model = MMDataParallel(model, device_ids=[0])
model.eval()

hook = ModelOutputSaveHook(save_interval=cfg.custom_hooks[0].save_interval, 
                            file_prefix=cfg.custom_hooks[0].file_prefix, 
                            path_config=cfg.custom_hooks[0].path_config, 
                            base_dir=cfg.custom_hooks[0].base_dir)
hook.run_for_sepcific_epoch(model, epoch=30)
