import os
import numpy as np

import torch

from pycocotools.coco import COCO

from mmcv import Config
from mmdet.apis import set_random_seed, init_detector

from mmdet.utils.lysto_utils import *

set_random_seed(0, deterministic=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

# define global variables
TAG = '[z-inference_lysto_12k]'
inference_configs = [
    # {'path_config': './configs/lyon/yolox_tiny_s1_lyon.py',
    #     'pr_curve_title': 'YoloX Tiny s1 | LYON', 'epoch': 15},
    {'path_config': './configs/lyon/retinanet_resnet50_s1_lyon.py',
        'pr_curve_title': 'RetinaNet ResNet50 s1 | LYON', 'epoch': 15, 'is_segmentation': False},
    {'path_config': './configs/lyon/fasterrcnn_resnet50_s1_lyon.py',
        'pr_curve_title': 'FasterRCNN s1 | LYON', 'epoch': 15, 'is_segmentation': False},
    {'path_config': './configs/lyon/scnet_resnet50_s1_lyon.py',
        'pr_curve_title': 'SCNet ResNet50 s1 | LYON', 'epoch': 15, 'is_segmentation': True},
    {'path_config': './configs/lyon/maskrcnn_resnet50_s1_lyon.py', 
        'pr_curve_title': 'MaskRCNN ResNet50 s1 | LYON', 'epoch': 15, 'is_segmentation': True},
    
    {'path_config': './configs/lyon/maskrcnn_lymphocytenet3_cm1_s6_lyon.py',
        'pr_curve_title': 'MaskRCNN LymphocyteNet s6 | LYON', 'epoch': 20, 'is_segmentation': True},
    {'path_config': './configs/lyon/maskrcnn_lymphocytenet3_cm1_s7_lyon.py',
        'pr_curve_title': 'MaskRCNN LymphocyteNet s7 | LYON', 'epoch': 20, 'is_segmentation': True},
    {'path_config': './configs/lyon/maskrcnn_lymphocytenet3_cm1_s14_lyon.py',
        'pr_curve_title': 'MaskRCNN LymphocyteNet s14 | LYON', 'epoch': 15, 'is_segmentation': True},
]

for inference_config in inference_configs:
    path_config = inference_config['path_config']
    pr_curve_title = inference_config['pr_curve_title']
    epoch = inference_config['epoch']
    is_segmentation = inference_config['is_segmentation']
    print('[path_config, pr_curve_title, epoch]', path_config, pr_curve_title, epoch)

    cfg = Config.fromfile(path_config)
    cfg.load_from = os.path.join(cfg.work_dir, f'epoch_{epoch}.pth')
    cfg.resume_from = ''
    print(TAG, '[cfg.load_from]', cfg.load_from)
    print(TAG, '[cfg.data.test.ann_file]', cfg.data.test.ann_file)
    
    PATH_INFERENCE_LYON = os.path.join(cfg.work_dir, 'inference_lyon')
    FILE_PREFIX = f'{cfg.MODEL_NAME}-s{cfg.S}-ep{epoch}'
    mmcv.mkdir_or_exist(PATH_INFERENCE_LYON)
    
    # load dataset (1)
    PATH_IMAGES1 = os.path.join(cfg.PATH_DATASET, 'test')
    PATH_IMAGES2 = os.path.join(cfg.PATH_DATASET, 'test2')
    LIST_DIR1 = os.listdir(PATH_IMAGES1)
    LIST_DIR2 = os.listdir(PATH_IMAGES2) if os.path.exists(PATH_IMAGES2) else []
    dataset = set(LIST_DIR1) | set(LIST_DIR2)
    # sort by name
    dataset = sorted(dataset, key=lambda x: int(x[:-4].split('_')[1]))
    # print(dataset)
    print(TAG, '[len(dataset)]', len(dataset))

    # load labels.csv
    labels_csv = pd.read_csv(os.path.join(cfg.PATH_DATASET, 'labels.csv'))
    labels_csv['organ_type'] = labels_csv.organ.apply(lambda x: x[:x.find('_')])
    # print(labels_csv.head(10))

    # load coco annotations
    coco_annotations = COCO(cfg.data.test.ann_file)
    map_imagename_2_cocoid = {obj['file_name']: obj['id'] for obj in coco_annotations.dataset['images']}

    # create model from config and load saved weights
    model = init_detector(cfg, cfg.load_from, device=device.type)
    model.CLASSES = cfg.classes

    # calculate model's output for all images in the dataset and save it as numpy array
    path_model_output = os.path.join(PATH_INFERENCE_LYON, f'outputs-{FILE_PREFIX}.npz')
    if not os.path.exists(path_model_output):
        outputs = calculate_model_outputs(model, dataset, [PATH_IMAGES1], [LIST_DIR1], setting=cfg.S, path_model_output=path_model_output)
    else:
        print(TAG, '[path_model_output already exists]', path_model_output)
        outputs = np.load(path_model_output, allow_pickle=True)
        outputs = outputs['arr_0']
    print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)

    # calculate per-threshold, per-image statistics
    path_statistics_csv = os.path.join(PATH_INFERENCE_LYON, f'statistics-{FILE_PREFIX}.csv')
    threshold_step = 0.01
    thresholds = np.arange(0, 1 + threshold_step, threshold_step)
    if not os.path.exists(path_statistics_csv):
        calculate_statistics_csv(outputs, dataset, coco_annotations, path_statistics_csv, thresholds, debug=False)
    else:
        print(TAG, '[path_statistics_csv already exists]', path_statistics_csv)

    # calculate and save PR-Curve using images-stats-per-threshold.csv
    path_pr_curve = os.path.join(PATH_INFERENCE_LYON, f'pr-curve-{FILE_PREFIX}.jpg')
    df = calculate_pr_curve(path_statistics_csv, thresholds, pr_curve_title, path_pr_curve)

    recalls, precisions = make_monotonic(df.recalls[::-1]), df.precisions[::-1]
    auc_pr_curve = auc(recalls, precisions)
    print(TAG, '[auc_pr_curve]', auc_pr_curve)

    # calculate and save centroids of model's output as csv file to use it in plotting function
    path_centroids_csv = os.path.join(PATH_INFERENCE_LYON, f'centroids-{FILE_PREFIX}.csv')
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

    path_outputs_drawn = os.path.join(cfg.work_dir, f'outputs-{FILE_PREFIX}')
    if not os.path.exists(path_outputs_drawn):
        os.mkdir(path_outputs_drawn)
        draw_outputs_on_images(outputs, dataset, coco_annotations, PATH_IMAGES1, path_outputs_drawn)
        print(TAG, '[len(path_outputs_drawn)]', len(os.listdir(path_outputs_drawn)))
    else:
        print(TAG, '[path_outputs_drawn already exists]', path_outputs_drawn)
