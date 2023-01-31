import os
import pandas as pd

import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel

from mmseg.apis import set_random_seed, init_segmentor
from mmseg.core.utils.lymph_count_eval_hook import LymphCountEvalHook

set_random_seed(0, deterministic=False)
MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')
opj = os.path.join
ope = os.path.exists

def perform_evaluation(cfg, eval_base_path, datatype, is_segmentation=True, fold=-1):
    """
    Args:
        cfg: mmdet Config
        eval_base_path: path to the eval csv file, this file will be created if evaluation was done during training as well.
        datatype: 
    """
    if not ope(eval_base_path):
        os.mkdir(eval_base_path)

    # find total epochs the model is trained upto
    epochs_trained = []
    for filename in os.listdir(cfg.work_dir):
        if filename.startswith('epoch_') and filename.endswith('.pth'):
            epochs_trained.append(int(filename[6:-4]))
    epochs_trained = set(sorted(epochs_trained))
    print(TAG, '[epochs_trained]', epochs_trained)

    # find the epochs on which model is already evaluated
    # and find the remaining epochs on which to evaluate
    eval_csv_name = f'{cfg.MODEL_NAME}-{datatype}.csv'
    path_eval_csv = opj(eval_base_path, eval_csv_name)
    if ope(path_eval_csv):
        df = pd.read_csv(path_eval_csv)
        epochs_evaluated = set(df.epoch.unique())
        epochs_to_evaluate = epochs_trained.difference(epochs_evaluated)
    else:
        epochs_to_evaluate = epochs_trained

    # epochs_to_evaluate = [30]
    # epochs_to_evaluate = set(range(2, 31, 2))
    epochs_to_evaluate = sorted(list(epochs_to_evaluate))
    print(TAG, '[epochs_to_evaluate]', epochs_to_evaluate)

    for epoch in epochs_to_evaluate:
        print(TAG, '[epoch]', epoch)
        cfg.load_from = opj(cfg.work_dir, f'epoch_{epoch}.pth')
        print(TAG, '[cfg.load_from]', cfg.load_from, ope(cfg.load_from))
        model = init_segmentor(cfg, cfg.load_from)
        model.CLASSES = cfg.classes
        model = MMDataParallel(model, device_ids=[0])
        model.eval()
        evaluator = LymphCountEvalHook(2, file_prefix=cfg.MODEL_NAME, path_config=cfg.PATH_CONFIG_FILE, base_dir=eval_base_path, fold=fold, debug=False)
        evaluator.eval_specific_epoch(model, epoch=epoch)

def eval_models(datatype, is_segmentation=True):
    """ This function evaluates all the trained models specified by the config file. """
    for path_config, eval_dir_name in configs_data:
        print(TAG, '[path_config]', path_config)
        cfg = Config.fromfile(path_config)
        print(TAG, '[cfg.work_dir]', cfg.work_dir)
        eval_base_path = opj(cfg.work_dir, eval_dir_name)
        perform_evaluation(cfg, eval_base_path, datatype, is_segmentation)

def eval_folds(datatype, is_segmentation=True):
    """ This function is same as eval_models, but for cross-validation models. """
    for path_config, eval_dir_name in configs_data:
        print(TAG, '[path_config]', path_config)
        for fold in range(1, 6):
            print(TAG, '[fold]', fold)
            cfg = Config.fromfile(path_config)
            # ./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/
            cfg.work_dir = opj(cfg.workdir, f'fold{fold}')
            print(TAG, '[cfg.work_dir]', cfg.work_dir)
            eval_base_path = opj(cfg.work_dir, eval_dir_name)
            perform_evaluation(cfg, eval_base_path, datatype, is_segmentation, fold)

TAG = '[z-evaluate_model]'
configs_data = [
    [opj(MMSEG_HOME_PATH, 'configs/lysto_hard/fcn_unet_s5_s1.py'), 'eval1_mask3_label'], 
]

eval_models('lysto')
# eval_folds('lysto')
