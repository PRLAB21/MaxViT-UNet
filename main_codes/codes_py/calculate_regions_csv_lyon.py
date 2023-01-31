import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb
from skimage.measure import label, regionprops
from tqdm import tqdm

import torch
import torch.nn.functional as F

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed, inference_detector, init_detector
from mmdet.core.utils import ml_metrics as metrics

set_random_seed(0, deterministic=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

def normalize_255(image):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    return image

def make_monotonic(array):
        monotonic_array = []
        previous_largest = -1
        for i, value in enumerate(array):
            if value > previous_largest:
                previous_largest = value
            monotonic_array.append(previous_largest)
        return monotonic_array

def ihc2dab(ihc_rgb):
    ihc_hed = rgb2hed(ihc_rgb)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    return normalize_255(ihc_d)
    # return ihc_h, ihc_e, ihc_d

def calculate_model_outputs(model, dataset, base_path, path_model_output):
    TAG = '[calculate_model_outputs]'
    # get model's prediction on all images
    outputs = []
    for image_idx, image_name in enumerate(mmcv.track_iter_progress(dataset)):
        # create image path
        img_path = os.path.join(base_path, image_name)
        # load image
        img_original = mmcv.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_original = ihc2dab(img_original)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        # get inference
        output = inference_detector(model.to(device), img_original)
        # convert float32 to float16
        output[0][0] = output[0][0].astype(np.float16)
        # convert python list to numpy array
        output[1][0] = np.array(output[1][0])
        # append this output
        outputs.append(output)
    # convert python list to numpy array
    outputs = np.asarray(outputs)
    # save the model's predictions as numpy compressed array
    np.savez_compressed(path_model_output, outputs)
    return outputs

def calculate_iou(gt_box, dt_box):
    # taken from: https://github.com/Treesfive/calculate-iou/blob/master/get_iou.py
    # 1.get the coordinate of intersection
    ixmin = max(dt_box[0], gt_box[0])
    ixmax = min(dt_box[2], gt_box[2])
    iymin = max(dt_box[1], gt_box[1])
    iymax = min(dt_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1, 0)
    ih = np.maximum(iymax - iymin + 1, 0)

    # 2. calculate the area of intersection
    intersection = iw * ih

    # 3. calculate the area of union
    union = ((dt_box[2] - dt_box[0] + 1) * (dt_box[3] - dt_box[1] + 1) + (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1) - intersection)

    # 4. calculate the overlaps between dt_box and gt_box
    iou = intersection / union

    return iou

def save_regions_csv(y_pred, dataset, base_path, file_prefix='regions'):
    TAG2 = TAG + '[save_regions_csv]'
    print(TAG2, '[y_pred]', y_pred.shape)
    # print(TAG2, '[images_name]', len(images_name), images_name[:5])
    result_analysis = []
    preds_ignore = []
    preds_overlap = []
    
    for i, name in tqdm(enumerate(dataset), total=len(dataset)):
    # for i in tqdm(range(len(y_pred)), total=len(y_pred)):
        mask = np.round(y_pred[i]).astype(int)
        label_mask = label(mask)
        regions = regionprops(label_mask)
        for props in regions:
            y0, x0 = props.centroid
            # area = props.area
            # e = props.eccentricity
            properties = [name, x0, y0]
            # if (e >= 1) or (area < 50):
                # preds_ignore.append(properties)
            # elif ((y0 < 31) | (y0 > 225)) | ((x0 < 31) | (x0 > 225)):
                # preds_overlap.append(properties)
            # else:
            result_analysis.append(properties)
    
    columns = ['image_id', 'x', 'y']
    df_result_analysis = pd.DataFrame(result_analysis, columns=columns)
    # df_preds_ignore = pd.DataFrame(preds_ignore, columns=columns)
    # df_preds_overlap = pd.DataFrame(preds_overlap, columns=columns)
    print(TAG2, '[df_result_analysis]', df_result_analysis.shape)
    # print(TAG2, '[df_preds_ignore]', df_preds_ignore.shape)
    # print(TAG2, '[df_preds_overlap]', df_preds_overlap.shape)
    df_result_analysis.to_csv(os.path.join(base_path, f'regions-{file_prefix}-result_analysis_lyon_lyonTest_4Aux_4_2.csv'), index=False)
    # df_preds_ignore.to_csv(os.path.join(base_path, f'regions-{file_prefix}-preds_ignore_lyon1.csv'), index=False)
    # df_preds_overlap.to_csv(os.path.join(base_path, f'regions-{file_prefix}-preds_overlap_lyon1.csv'), index=False)

# define global variables
TAG = '[z-final_model_script]'
path_configs = [
    './configs/lyon/maskrcnn_lymphofusion3_s1_lyon.py',
   # './configs/lyon/maskrcnn_lymphofusion2_s1_lyon.py',
]

pr_curve_titles = [
    #'Deep_Fusion_Block | LYON',
    'attetntion_3x3_FusionDeep_4AuxilaryChannels | LYON',
]

EPOCH = 30
for path_config, pr_curve_title in zip(path_configs, pr_curve_titles):
    cfg = Config.fromfile(path_config)
    cfg.load_from = os.path.join(cfg.work_dir, f'epoch_{EPOCH}.pth')
    print(TAG, '[cfg.load_from]', cfg.load_from)
    cfg.resume_from = ''

    # load dataset

    # PATH_IMAGES = './lymphocyte_dataset/LYON-dataset/lyon_patch_overlap_onboundries_splits'
    # PATH_IMAGES = '/home/gpu02/lyon_dataset/lyon_patch_overlap_onboundries_splits/lyon_patch_overlap_onboundries-split2'
    PATH_IMAGES = '/home/gpu02/lyon_dataset/lyon_patch_overlap_onboundries_splits/split_4_2'
    print(TAG, '[PATH_IMAGES]', PATH_IMAGES)
    dataset = os.listdir(PATH_IMAGES)
    print(TAG, '[len(dataset)]', len(dataset), dataset[:5])
    # sort by name
    #dataset = sorted(dataset, key=lambda x: (int(x[4:-4].split('_')[0]), int(x[4:-4].split('_')[1])))
    print(TAG, '[len(dataset)]', len(dataset), dataset[:5])

    # calculate model's output for all images in the dataset and save it as numpy array
    path_model_output = os.path.join(cfg.work_dir, f'outputs-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}_lyonTest_4Aux_4_2.npz')
    if not os.path.exists(path_model_output):
        # create model from config and load saved weights
        model = init_detector(cfg, cfg.load_from, device=device.type)
        model.CLASSES = cfg.classes
        outputs = calculate_model_outputs(model, dataset, PATH_IMAGES, path_model_output=path_model_output)
    else:
        print(TAG, '[path_model_output already exists]', path_model_output)
        outputs = np.load(path_model_output, allow_pickle=True)
        outputs = outputs['arr_0']
    print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)

    regions_csv_path = os.path.join(cfg.work_dir, f'regions_csvs')
    os.makedirs(regions_csv_path, exist_ok=True)
    masks = outputs[:, 1, 0]
    print(TAG, '[masks]', masks.shape, masks.dtype, masks[0].shape)

    # view masks per image separately
    # for i, y_pred in enumerate(masks[:10]):
    #     for j, mask in enumerate(y_pred):
    #         print(TAG, i, j)
    #         mask = mask[..., None].astype(np.uint8)
    #         mask[mask == 1] = 255
    #         cv2.imshow('mask', mask)
    #         cv2.waitKey(100)

    y_preds = []
    # view masks per image combined
    for i, y_pred in enumerate(masks):
        mask = y_pred.sum(axis=0)
        mask = mask.astype(np.uint8)
        print(y_pred.shape, mask.shape)
        if mask.shape:
            mask[mask > 0] = 255
            y_preds.append(mask)
        else:
            y_preds.append(np.zeros((256, 256)))
        # cv2.imshow('mask', mask)
        # cv2.waitKey(100)
    y_preds = np.array(y_preds)
    print(TAG, y_preds.shape)
    save_regions_csv(y_preds, dataset, base_path=regions_csv_path)
    
