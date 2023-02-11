import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import auc

from pycocotools.coco import COCO

from mmcv import Config
from mmseg.apis import set_random_seed, init_detector

from mmseg.utils.lysto_utils import *

MMSEG_HOME_PATH = '/home/zunaira/maskrcnn-lymphocyte-detection/mmdetection'
set_random_seed(0, deterministic=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')


def plot_centroids_randomly(images_name, path_centroids_csv):
    # plot centroids on randomly selected image
    df_centroids_info = pd.read_csv(path_centroids_csv)
    image_index = np.random.randint(0, len(images_name))
    if images_name[image_index] in LIST_DIR1:
        image_path = opj(PATH_IMAGES1, images_name[image_index])
    else:
        image_path = opj(PATH_IMAGES2, images_name[image_index])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ihc2dab(image)
    predictions = df_centroids_info[df_centroids_info.image_name == images_name[image_index]]
    # print('predictions:', predictions)
    plt.figure(figsize=(10, 10))
    for i, prediction in predictions.iterrows():
        center_x, center_y = int(prediction.center_x), int(prediction.center_y)
        cv2.ellipse(image, (center_x, center_y), (13, 13), 0, 0, 360, (0, 0, 255), 1)
    plt.imshow(image)


def _infer(cfg, output_dir, images_name, base_path_dataset, epoch, pr_curve_title='', 
            is_segmentation=True, coco_annotations=None, labels_csv=None, 
            flag_pr_curve=True, flag_centroids=True, flag_draw_output=True):

    cfg.load_from = opj(MMSEG_HOME_PATH, cfg.work_dir, f'epoch_{epoch}.pth')
    print(TAG, '[cfg.load_from]', cfg.load_from)

    path_output_one_stage = opj(MMSEG_HOME_PATH, cfg.work_dir, output_dir)
    output_prefix = f'{cfg.MODEL_NAME}-s{cfg.S}-ep{epoch}'
    # output_prefix = f'{cfg.MODEL_NAME}-s{cfg.S}'
    print(TAG, '[path_output_one_stage]', path_output_one_stage)
    if not ope(path_output_one_stage):
        os.mkdir(path_output_one_stage)

    # create model from config and load saved weights
    model = init_detector(cfg, cfg.load_from, device=device.type)
    model.CLASSES = cfg.classes


    # calculate model's output for all images in the images_name and save it as numpy array
    path_model_output = opj(path_output_one_stage, f'outputs-{output_prefix}.npz')
    print(TAG, '[path_model_output]', path_model_output)
    if not ope(path_model_output):
        outputs = calculate_model_outputs(model, images_name, base_path_dataset, path_model_output, is_segmentation=is_segmentation)
    else:
        print(TAG, '[path_model_output already exists]', path_model_output)
        outputs = np.load(path_model_output, allow_pickle=True)
        outputs = outputs['arr_0']
        # np.savez_compressed(opj(path_output_one_stage, f'outputs-{output_prefix}-temp.npz'), outputs[:100])
    print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)


    # calculate per-threshold, per-image statistics
    path_statistics_csv = opj(path_output_one_stage, f'statistics-{output_prefix}.csv')
    print(TAG, '[path_statistics_csv]', path_statistics_csv)
    threshold_step = 0.01
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    if not ope(path_statistics_csv):
        calculate_statistics_csv(outputs, images_name, path_statistics_csv, thresholds, coco_annotations=coco_annotations, is_segmentation=is_segmentation, debug=False)
    else:
        print(TAG, '[path_statistics_csv already exists]', path_statistics_csv)


    if flag_pr_curve:
        # calculate and save PR-Curve using images-stats-per-threshold.csv
        path_pr_curve = opj(path_output_one_stage, f'pr_curve-{output_prefix}')
        df = calculate_pr_curve(path_statistics_csv, thresholds, pr_curve_title, path_pr_curve)
        recalls, precisions = make_monotonic(df.recalls[::-1]), df.precisions[::-1]
        auc_pr_curve = auc(recalls, precisions)
        print(TAG, '[auc_pr_curve]', auc_pr_curve)


    if flag_centroids:
        # calculate and save centroids of model's output as csv file to use it in plotting function
        path_centroids_csv = opj(path_output_one_stage, f'centroids-{output_prefix}.csv')
        print('[path_centroids_csv]', path_centroids_csv)
        if not ope(path_centroids_csv):
            calculate_centroids_csv(outputs, images_name, path_centroids_csv, labels_csv=labels_csv, is_segmentation=is_segmentation)
        else:
            print(TAG, '[path_centroids_csv already exists]', path_centroids_csv)


    if flag_draw_output:
        # draw output in images along with ground truth
        path_outputs_drawn = opj(path_output_one_stage, f'outputs_drawn-{output_prefix}')
        if not ope(path_outputs_drawn):
            os.mkdir(path_outputs_drawn)
            if coco_annotations:
                draw_outputs_with_original(outputs, images_name, base_path_dataset, path_outputs_drawn, coco_annotations)
            else:
                draw_outputs_without_original(outputs, images_name, base_path_dataset, path_outputs_drawn)
            print(TAG, '[len(path_outputs_drawn)]', len(os.listdir(path_outputs_drawn)))
        else:
            print(TAG, '[path_outputs_drawn already exists]', path_outputs_drawn)


def infer_lysto_testset(inference_configs, internal_images_dir):
    """
    This function performs inference on LYSTO internal images.
    Args:
        internal_set (str): could be any of train/val/test
    """
    for inference_config in inference_configs:
        path_config = inference_config['path_config']
        dataset_type = inference_config['dataset_type']
        pr_curve_title = inference_config['pr_curve_title']
        epoch = inference_config['epoch']
        is_segmentation = inference_config['is_segmentation']
        print('[path_config, dataset_type, pr_curve_title, epoch, is_segmentation]', path_config, dataset_type, pr_curve_title, epoch, is_segmentation)

        cfg = Config.fromfile(path_config)

        # load images name
        base_path_dataset = opj(cfg.PATH_DATASET, internal_images_dir)
        print(TAG, '[base_path_dataset]', base_path_dataset)
        images_name = os.listdir(base_path_dataset)
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])
        # sort by name
        images_name = sorted(images_name, key=lambda x: int(x[9:-4]))
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])

        # load coco annotations
        coco_annotations = COCO(cfg.data.test.ann_file)

        # load labels.csv
        labels_csv = pd.read_csv(opj(cfg.PATH_DATASET, 'labels.csv'))
        labels_csv['organ_type'] = labels_csv.organ.apply(lambda x: x[:x.find('_')])

        _infer(cfg=cfg, output_dir='infer_lysto_testset', images_name=images_name, 
                epoch=epoch, pr_curve_title=pr_curve_title, is_segmentation=is_segmentation, 
                coco_annotations=coco_annotations, labels_csv=labels_csv)


def infer_lysto_12k(inference_configs, external_images_dir):
    """
    This function performs inference on 12,000 images in LYSTO external test set.
    """
    for inference_config in inference_configs:
        path_config = inference_config['path_config']
        dataset_type = inference_config['dataset_type']
        pr_curve_title = inference_config['pr_curve_title']
        epoch = inference_config['epoch']
        is_segmentation = inference_config['is_segmentation']
        print('[path_config, dataset_type, pr_curve_title, epoch, is_segmentation]', path_config, dataset_type, pr_curve_title, epoch, is_segmentation)

        cfg = Config.fromfile(path_config)

        # load images_name
        base_path_dataset = opj(cfg.PATH_DATASET, external_images_dir)
        print(TAG, '[base_path_dataset]', base_path_dataset)
        images_name = os.listdir(base_path_dataset)
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])
        # sort by name
        images_name = sorted(images_name, key=lambda x: int(x[9:-4]))
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])

        _infer(cfg=cfg, output_dir='infer_lysto_testset', images_name=images_name, 
                epoch=epoch, pr_curve_title=pr_curve_title, is_segmentation=is_segmentation, 
                coco_annotations=None, labels_csv=None)


def infer_lyon_testset(inference_configs, internal_images_dir):
    """
    This function performs inference on LYON internal images.
    Args:
        internal_set (str): could be any of train/val/test
    """
    for inference_config in inference_configs:
        path_config = inference_config['path_config']
        dataset_type = inference_config['dataset_type']
        pr_curve_title = inference_config['pr_curve_title']
        epoch = inference_config['epoch']
        is_segmentation = inference_config['is_segmentation']
        print('[path_config, pr_curve_title, epoch, is_segmentation]', path_config, pr_curve_title, epoch, is_segmentation)

        cfg = Config.fromfile(path_config)

        # load images name
        base_path_dataset = opj(MMSEG_HOME_PATH, cfg.PATH_DATASET, internal_images_dir)
        print(TAG, '[base_path_dataset]', base_path_dataset)
        images_name = os.listdir(base_path_dataset)
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])
        # sort by name
        images_name = sorted(images_name, key=lambda x: (int(x[4:-4].split('_')[0]), int(x[4:-4].split('_')[1])))
        print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])

        # load coco annotations
        coco_annotations = COCO(opj(MMSEG_HOME_PATH, cfg.data.test.ann_file))

        _infer(cfg=cfg, output_dir='infer_lyon_testset', images_name=images_name, base_path_dataset=base_path_dataset, 
                epoch=epoch, pr_curve_title=pr_curve_title, is_segmentation=is_segmentation, 
                coco_annotations=coco_annotations, labels_csv=None, flag_draw_output=False)


def infer_lyon_160k(inference_configs):
    """
    This function performs inference on 160,000 images in LYON external test set.
    """
    for inference_config in inference_configs:
        path_config = inference_config['path_config']
        epoch = inference_config['epoch']
        pr_curve_title = inference_config['pr_curve_title']
        is_segmentation = inference_config['is_segmentation']
        print('[path_config, pr_curve_title, epoch]', path_config, pr_curve_title, epoch)

        for split in range(1, 11):
            cfg = Config.fromfile(path_config)

            # load images name
            base_path_images = f'/home/zunaira/lyon_dataset/lyon_patch_overlap_onboundries_splits/lyon_patch_overlap_onboundries-split{split}'
            print(TAG, '[base_path_images]', base_path_images)
            images_name = os.listdir(base_path_images)
            print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])
            # sort by name
            images_name = sorted(images_name, key=lambda x: (int(x[4:-4].split('_')[0]), int(x[4:-4].split('_')[1])))
            # images_name = sorted(images_name, key=lambda x: (int(x[4:-4].split('-')[0]), int(x[4:-4].split('-')[1])))
            print(TAG, '[len(images_name)]', len(images_name), images_name[:5], images_name[-5:])

            _infer(cfg=cfg, output_dir='infer_lyon_160k', images_name=images_name, 
                    epoch=epoch, pr_curve_title=pr_curve_title, is_segmentation=is_segmentation, 
                    coco_annotations=None, labels_csv=None)


def infer_cross_validation():
    # define global variables
    TAG = '[z-final_model_script]'
    path_configs = [
        '../../configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s10_lysto.py',
    ]

    pr_curve_titles = [
        'MaskRCNN LymphocyteNet s10 (Proposed) | LYSTO',
    ]

    EPOCH = 10
    for path_config, pr_curve_title in zip(path_configs, pr_curve_titles):
        print('[path_config, pr_curve_title]', path_config, pr_curve_title)
        for fold in range(1, 11):
            print('[fold]', fold)
            cfg = Config.fromfile(path_config)
            cfg.work_dir = f'./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/fold{fold}/'
            cfg.load_from = os.path.join(cfg.work_dir, f'epoch_{EPOCH}.pth')
            cfg.data.val.ann_file = f'{cfg.PATH_DATASET}/cross-validation-folds-coco-json/fold_{fold}_val.json'
            print(TAG, '[cfg.work_dir]', cfg.work_dir)
            print(TAG, '[cfg.load_from]', cfg.load_from)
            print(TAG, '[cfg.data.test.ann_file]', cfg.data.val.ann_file)
            cfg.resume_from = ''
            # print(cfg.pretty_text)

            PATH_FINAL_MODEL_SCRIPT = os.path.join(cfg.work_dir, 'final_model_script')
            if not os.path.exists(PATH_FINAL_MODEL_SCRIPT):
                os.mkdir(PATH_FINAL_MODEL_SCRIPT)

            # load dataset (1)
            base_path_IHC = os.path.join(cfg.PATH_DATASET, 'train_val_IHC')
            dataset = pd.read_csv(os.path.join(cfg.PATH_DATASET, f'cross-validation-folds/fold_{fold}_val.csv')).x.values.tolist()
            # sort by name
            dataset = sorted(dataset, key=lambda x: int(x[:-4].split('_')[1]))
            # print(dataset)
            print(TAG, '[len(dataset)]', len(dataset))

            # load labels.csv
            labels_csv = pd.read_csv(os.path.join(cfg.PATH_DATASET, 'labels.csv'))
            labels_csv['organ_type'] = labels_csv.organ.apply(lambda x: x[:x.find('_')])
            # print(labels_csv.head(10))

            # load coco annotations
            coco_annotations = COCO(cfg.data.val.ann_file)

            # create model from config and load saved weights
            model = init_detector(cfg, cfg.load_from, device=device.type)
            model.CLASSES = cfg.classes

            # calculate model's output for all images in the dataset and save it as numpy array
            path_model_output = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'outputs-12k-{cfg.MODEL_NAME}-s{cfg.S}-f{fold}.npz')
            print('[path_model_output]', path_model_output)
            if not os.path.exists(path_model_output):
                outputs = calculate_model_outputs(model, dataset, base_path_IHC, path_model_output)
            else:
                print(TAG, '[path_model_output already exists]', path_model_output)
                outputs = np.load(path_model_output, allow_pickle=True)
                outputs = outputs['arr_0']
            print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)

            # calculate per-threshold, per-image statistics
            # path_statistics_csv = os.path.join(cfg.work_dir, f'statistics-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}.csv')
            path_statistics_csv = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'statistics-{cfg.MODEL_NAME}-s{cfg.S}-f{fold}.csv')
            print('[path_statistics_csv]', path_statistics_csv)
            threshold_step = 0.01
            thresholds = np.arange(0, 1 + threshold_step, threshold_step)
            if not os.path.exists(path_statistics_csv):
                calculate_statistics_csv(outputs, dataset, coco_annotations, path_statistics_csv, thresholds, debug=False)
            else:
                print(TAG, '[path_statistics_csv already exists]', path_statistics_csv)

            # calculate and save PR-Curve using images-stats-per-threshold.csv
            # path_pr_curve = os.path.join(cfg.work_dir, f'pr-curve-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}.jpg')
            path_pr_curve = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'pr_curve-{cfg.MODEL_NAME}-s{cfg.S}-f{fold}.jpg')
            print('[path_pr_curve]', path_pr_curve)
            df = calculate_pr_curve(path_statistics_csv, thresholds, pr_curve_title, path_pr_curve)

            recalls, precisions = make_monotonic(df.recalls[::-1]), df.precisions[::-1]
            auc_pr_curve = auc(recalls, precisions)
            print(TAG, '[auc_pr_curve]', auc_pr_curve)

            # calculate and save centroids of model's output as csv file to use it in plotting function
            # path_centroids_csv = os.path.join(cfg.work_dir, f'centroids-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}.csv')
            path_centroids_csv = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'centroids-{cfg.MODEL_NAME}-s{cfg.S}-f{fold}.csv')
            print('[path_centroids_csv]', path_centroids_csv)
            if not os.path.exists(path_centroids_csv):
                calculate_centroids_csv(outputs, dataset, path_centroids_csv, labels_csv=labels_csv)
            else:
                print(TAG, '[path_centroids_csv already exists]', path_centroids_csv)

            # plot centroids on randomly selected image
            df_centroids_info = pd.read_csv(path_centroids_csv)
            image_index = np.random.randint(0, len(dataset))
            if dataset[image_index] in LIST_DIR1:
                image_path = os.path.join(PATH_IMAGES1, dataset[image_index])
            else:
                image_path = os.path.join(PATH_IMAGES2, dataset[image_index])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ihc2dab(image)
            predictions = df_centroids_info[df_centroids_info.image_name == dataset[image_index]]
            # print('predictions:', predictions)
            plt.figure(figsize=(10, 10))
            for i, prediction in predictions.iterrows():
                center_x, center_y = int(prediction.center_x), int(prediction.center_y)
                cv2.ellipse(image, (center_x, center_y), (13, 13), 0, 0, 360, (0, 0, 255), 1)
            plt.imshow(image)

            path_outputs_drawn = os.path.join(cfg.work_dir, f'outputs-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}')
            if not os.path.exists(path_outputs_drawn):
                os.mkdir(path_outputs_drawn)
                draw_outputs_on_images(outputs, dataset, coco_annotations, PATH_IMAGES1, path_outputs_drawn)
                print(TAG, '[len(path_outputs_drawn)]', len(os.listdir(path_outputs_drawn)))
            else:
                print(TAG, '[path_outputs_drawn already exists]', path_outputs_drawn)

            print('-' * 50)


####### infer one stage #######
# algorithm: for each config
# - load config
# - load images_name
# - load model
# - inference models on images one by one and save as npz file
# - using inference npz save count csv
# - draw model output on imagse one by one and save image by image

# cases:
# - lysto internal-test
# - lysto 12k images (1 split)
# - lyon internal-test
# - lyon 160k images (10 splits)
# - cross-validation folds

# define global variables
TAG = '[z-infer_one_stage]'
opj = os.path.join
ope = os.path.exists
inference_configs = [
    # {'path_config': '../../configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto_combined.py', 
    #     'pr_curve_title': 'MaskRCNN LymphocyteNet s13 | LYSTO', 
    #     'dataset_type': 'lysto', 'epoch': 10, 'is_segmentation': False},
    # {'path_config': '../../configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s14_lysto_combined.py', 
    #     'pr_curve_title': 'MaskRCNN LymphocyteNet s14 | LYSTO', 
    #     'dataset_type': 'lysto', 'epoch': 10, 'is_segmentation': False},

    # {'path_config': '../../configs/lyon/yolox_s_s1_lyon.py', 
    #     'pr_curve_title': 'YoloX-S (Pre-trained) | LYON', 
    #     'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': False},
    # {'path_config': '../../configs/lyon/yolox_s_s2_lyon.py', 
    #     'pr_curve_title': 'YoloX-S (Scratch) | LYON', 
    #     'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': False},
    {'path_config': '../../configs/lyon/retinanet_resnet50_s1_lyon.py', 
        'pr_curve_title': 'RetinaNet ResNet50 (Pre-trained) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': False},
    {'path_config': '../../configs/lyon/fasterrcnn_resnet50_s1_lyon.py', 
        'pr_curve_title': 'FasterRCNN (Pre-trained) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': False},
    {'path_config': '../../configs/lyon/scnet_resnet50_s1_lyon.py', 
        'pr_curve_title': 'SCNet ResNet50 | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
    {'path_config': '../../configs/lyon/maskrcnn_resnet50_s1_lyon.py', 
        'pr_curve_title': 'MaskRCNN ResNet50 (s1) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
    {'path_config': '../../configs/lyon/maskrcnn_resnet50_s2_lyon.py', 
        'pr_curve_title': 'MaskRCNN ResNet50 (s2) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
    {'path_config': '../../configs/lyon/maskrcnn_resnet50_s3_lyon.py', 
        'pr_curve_title': 'MaskRCNN ResNet50 (s3) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
    {'path_config': '../../configs/lyon/maskrcnn_resnet50_s4_lyon.py', 
        'pr_curve_title': 'MaskRCNN ResNet50 (s4) | LYON', 
        'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
    {'path_config': '../../configs/lyon/maskrcnn_lymphocytenet3_cm1_s6_lyon.py', 
        'pr_curve_title': 'MaskRCNN LymphocyteNet s6 | LYON', 
        'dataset_type': 'lyon', 'epoch': 10, 'is_segmentation': True},
    # {'path_config': '../../configs/lyon/maskrcnn_lymphocytenet3_cm1_s7_lyon.py', 
    #     'pr_curve_title': 'MaskRCNN LymphocyteNet s7 | LYON', 
    #     'dataset_type': 'lyon', 'epoch': 20, 'is_segmentation': True},
    # {'path_config': '../../configs/lyon/maskrcnn_lymphocytenet3_cm1_s14_lyon.py', 
    #     'pr_curve_title': 'MaskRCNN LymphocyteNet s14 | LYON', 
    #     'dataset_type': 'lyon', 'epoch': 15, 'is_segmentation': True},
]

infer_lyon_testset(inference_configs, 'Validation')
