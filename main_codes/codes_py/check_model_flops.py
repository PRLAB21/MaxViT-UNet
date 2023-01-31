import os
import torch
from mmcv import Config
from mmcls.models import build_classifier
from mmcv.cnn import get_model_complexity_info

MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmclassification'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'

def main_get_flops(config, input_shape):
    TAG = '[main_get_flops]'
    print(TAG, '[starts]')

    cfg = Config.fromfile(config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_classifier(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError('FLOPs counter is currently not currently supported with {}'.format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print()
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    print(TAG, '[ends]')

config_file_path = os.path.join(MMSEG_HOME_PATH, 'configs/lysto_lymph_vs_nolymph/lymphnet_s1.py')
main_get_flops(config_file_path, (3, 224, 224))
