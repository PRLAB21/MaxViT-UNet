import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from skimage.color import rgb2hed, hed2rgb

import torch
import torch.nn.functional as F

import mmcv
from mmcv import Config
from mmseg.apis import set_random_seed, inference_segmentor, init_segmentor
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

def get_contours_from_mask(mask_image_path=None, mask_image=None):
    if mask_image_path is not None:
        # print('[mask_image_path]', mask_image_path)
        im = cv2.imread(mask_image_path)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if len(im.shape) > 2 else im
    elif mask_image is not None:
        imgray = mask_image
    # print('[imgray]', imgray.dtype, imgray.shape, imgray.min(), imgray.max())
    imgray = normalize_255(imgray)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    # print('[thresh]', thresh.dtype, thresh.shape, thresh.min(), thresh.max())
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours.reshape(-1, 1, 2)
    # print('[contours]', len(contours), contours[0].shape)
    return contours, len(contours)

def get_bboxes_from_contours(contours):
    bboxes = []
    for contour in contours:
        if len(contour) > 3:
            # print(contour, contour.shape)
            x_min = np.min(contour[:, 0, 0])
            x_max = np.max(contour[:, 0, 0])
            y_min = np.min(contour[:, 0, 1])
            y_max = np.max(contour[:, 0, 1])
            bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
    bboxes = np.array(bboxes)
    # print('[bboxes]', bboxes)
    return bboxes

def calculate_model_outputs(model, dataset, base_path_images, list_images_name, setting, path_model_output):
    TAG = '[calculate_model_outputs]'
    # get model's prediction on all images
    outputs = []
    for image_idx, image_name in enumerate(mmcv.track_iter_progress(dataset)):
        # create image path based on folder
        for list_idx in range(len(list_images_name)):
            if image_name in list_images_name[list_idx]:
                break
        # create image path
        img_path = os.path.join(base_path_images[list_idx], image_name)
        # load image
        img_original = mmcv.imread(img_path)
        # get inference
        output = inference_segmentor(model, img_original)
        # print('[output]', type(output), len(output), type(output[0]), output[0].dtype, output[0].shape)
        output[0] = output[0].astype(np.uint8)
        # print('[output]', type(output), len(output), type(output[0]), output[0].dtype, output[0].shape)
        # append this output
        outputs.append(output)
        # if image_idx > 19: break
    
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

def draw_outputs_on_images(outputs, dataset, coco_annotations, base_path_images, path_outputs_drawn):
    print(TAG, '[draw_outputs_on_images][1]')
    distance = 12
    threshold = 0.5
    font_size = 0.5
    font_weight = 1
    resize_scale = 2

    for idx, (image_name, output) in tqdm(enumerate(zip(dataset, outputs)), total=len(dataset)):
        # get gt_bboxes
        annotation_ids = coco_annotations.getAnnIds(imgIds=[map_imagename_2_cocoid[image_name]])
        gt_bboxes = np.array([coco_annotations.anns[i]['bbox'] for i in annotation_ids])
        gt_bboxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in gt_bboxes])

        # get dt_bboxes
        dt_bboxes = output[0][0]
        scores = dt_bboxes[:, 4]
        # dt_bboxes = dt_bboxes[:, :]
        dt_bboxes = dt_bboxes[np.where(scores > threshold)[0]]    
        dt_count1 = int(len(dt_bboxes))
        centroid_x = (dt_bboxes[:, 0] + dt_bboxes[:, 2]) / 2
        centroid_y = (dt_bboxes[:, 1] + dt_bboxes[:, 3]) / 2
        merged = []
        for j in range(0, dt_count1 - 1):
            if j not in merged:
                for k in range(j + 1, dt_count1):
                    diff_x = np.float32(np.square(centroid_x[j] - centroid_x[k]))
                    diff_y = np.float32(np.square(centroid_y[j] - centroid_y[k]))
                    distance = int(np.sqrt(diff_x + diff_y))
                    if distance <= 13:
                        merged.append(k)
        dt_bboxes = np.delete(dt_bboxes, merged, axis=0)

        # load image
        img_path = os.path.join(base_path_images, image_name)
        image = cv2.imread(img_path)
        image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)

        # calculate IOUs between each gt_bbox and dt_bbox
        count_gt_bboxes = len(gt_bboxes)
        count_dt_bboxes = len(dt_bboxes)
        iou_matrix = np.zeros((count_gt_bboxes, count_dt_bboxes))
        for i in range(count_gt_bboxes):
            for j in range(count_dt_bboxes):
                iou_matrix[i, j] = calculate_iou(gt_bboxes[i], dt_bboxes[j])

        for i, bboxes in enumerate([gt_bboxes, dt_bboxes]):
            for j, bbox in enumerate(bboxes):
                if i == 0:
                    # false negative
                    if sum(iou_matrix[j, :]) == 0:
                        score_text = "FN"
                        bg_color = (0, 255, 255) if i == 0 else ()
                        text_color = (0, 0, 0)
                    else: continue
                else:
                    # false positive
                    if sum(iou_matrix[:, j]) == 0:
                        score_text = f"FP({int(100 * bbox[4])}%)"
                        bg_color = (0, 0, 255)
                    # true positive
                    elif sum(iou_matrix[:, j]) > 0:
                        score_text = f"TP({int(100 * bbox[4])}%)"
                        bg_color = (0, 128, 0)
                    else: continue
                    text_color = (255, 255, 255)

                # get coordinates of bbox
                x1 = int(bbox[0] * resize_scale)
                y1 = int(bbox[1] * resize_scale)
                x2 = int(bbox[2] * resize_scale)
                y2 = int(bbox[3] * resize_scale)

                # skip very small bboxes
                if np.abs(x1 - x2) < 3 * resize_scale or np.abs(y1 - y2) < 3 * resize_scale: continue

                # draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, 2, cv2.LINE_AA)

                # in order to draw text within image, recalculate coordinates
                x2 = x1 + 30 if i == 0 else x1 + 75
                y2 = y1 - 25
                if x2 >= image.shape[1]:
                    x2 = int(bbox[2]) * resize_scale
                    x1 = x2 - 25 if i == 0 else x2 - 75
                if y2 <= 0:
                    y2 = int(bbox[3]) * resize_scale
                    y1 = y2 + 25
                # draw rectangle for text bg
                cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
                # draw text
                cv2.putText(image, score_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_weight, cv2.LINE_AA)

        filename = os.path.join(path_outputs_drawn, image_name)
        cv2.imwrite(filename, image)

def calculate_centroids_csv(outputs, dataset, path_centroids_csv, labels_csv=None, debug=False):
    columns = ['id', 'image_name', 'center_x', 'center_y', 'x1', 'y1', 'x2', 'y2']
    if labels_csv is not None:
        columns += ['organ', 'organ_type']
    print(TAG, '[columns]', columns)
    centroids_info = {key: [] for key in columns}
    threshold = 0.5
    
    # iterate over all dataset images
    for id, image_name in enumerate(mmcv.track_iter_progress(dataset)):
        image_name = image_name[:-4] + '.png'
        row = labels_csv[labels_csv.x == image_name] if labels_csv is not None else None
        
        # get model's prediction
        output = outputs[id]
        
        # get classification scores (confidence)
        scores = output[0][0][:, 4]
        
        # get bounding boxes
        bboxes = output[0][0][:, :4]
        
        # filter boxes with confidence more then threshold
        bboxes = bboxes[np.where(scores > threshold)[0]]
        
        # append data of filtered bboxes in dictionary
        for box in bboxes:
            centroids_info['id'].append(id)
            centroids_info['image_name'].append(image_name)
            centroids_info['center_x'].append((box[0] + box[2]) / 2)
            centroids_info['center_y'].append((box[1] + box[3]) / 2)
            centroids_info['x1'].append(box[0])
            centroids_info['y1'].append(box[1])
            centroids_info['x2'].append(box[2])
            centroids_info['y2'].append(box[3])
            if row is not None:
                centroids_info['organ'].append(row.organ.values[0])
                centroids_info['organ_type'].append(row.organ_type.values[0])

    df_centroids_info = pd.DataFrame(centroids_info)
    df_centroids_info.to_csv(path_centroids_csv, index=False)
    print(TAG, f'\ncsv saved at {path_centroids_csv}\n')

def calculate_statistics_csv(cfg, outputs, dataset, path_statistics_csv, debug=False):
    columns = ['distance', 'image_name', 'gt_count', 'dt_count', 'TP', 'FP', 'FN']
    image_info = {key: [] for key in columns}

    # iterate over all dataset images
    for d in [0, 16]:
        print(TAG, f'running loop for d = {d}')
        for id, image_name in enumerate(dataset):
            # store info
            image_info['distance'].append(d)
            image_info['image_name'].append(image_name)
            # get ground truth count
            gt_contours, gt_count = get_contours_from_mask(mask_image_path=os.path.join(cfg.PATH_DATASET, cfg.data.test.ann_dir, image_name))
            image_info['gt_count'].append(gt_count)
            # get detection/predicted count
            dt_contours, dt_count = get_contours_from_mask(mask_image=outputs[id, 0])
            dt_bboxes = get_bboxes_from_contours(dt_contours)
            dt_count = len(dt_bboxes)
            
            merged = []
            if dt_count > 0:
                # get centroid (x, y) of remaining bounding boxes
                centroid_x = (dt_bboxes[:, 0] + dt_bboxes[:, 2]) / 2
                centroid_y = (dt_bboxes[:, 1] + dt_bboxes[:, 3]) / 2
                if d > 0:
                    for j in range(0, dt_count - 1):
                        if j not in merged:
                            for k in range(j + 1, dt_count):
                                diff_x = np.square(np.float32(centroid_x[j] - centroid_x[k]))
                                diff_y = np.square(np.float32(centroid_y[j] - centroid_y[k]))
                                distance = int(np.sqrt(diff_x + diff_y))
                                if distance <= d:
                                    merged.append(k)
            
            # get new filtered detection count
            dt_count = int(dt_count - len(merged))
            image_info['dt_count'].append(dt_count)

            confusion_matrix = np.array([[0, 0], [0, 0]])
            confusion_matrix[0][0] += min(gt_count, dt_count)
            confusion_matrix[0][1] += np.abs(gt_count - dt_count) if gt_count < dt_count else 0
            confusion_matrix[1][0] += np.abs(gt_count - dt_count) if gt_count > dt_count else 0

            # calculate TP, FP, FN for ith image
            TP = confusion_matrix[0][0]
            FP = confusion_matrix[0][1]
            FN = confusion_matrix[1][0]
            # TN = confusion_matrix[1][1]
            # P = TP / (TP + FP)
            # R = TP / (TP + FN)
            image_info['TP'].append(TP)
            image_info['FP'].append(FP)
            image_info['FN'].append(FN)
            # image_info['TN'].append(TN)

        df_image_info = pd.DataFrame(image_info)
        df_image_info.to_csv(path_statistics_csv, index=False)
        print(TAG, f'\ncsv saved at {path_statistics_csv}\n')

def calculate_counts_csv(outputs, dataset, path_counts_csv, debug=False):
    columns = ['id', 'count']
    image_info = {key: [] for key in columns}
    d = 0

    # iterate over all dataset images
    for id, image_name in tqdm(enumerate(dataset), total=len(dataset)):
        # store info
        image_info['id'].append(int(image_name[5:-4]))

        # get detection/predicted count
        dt_contours, dt_count = get_contours_from_mask(mask_image=outputs[id, 0])
        dt_bboxes = get_bboxes_from_contours(dt_contours)
        dt_count = len(dt_bboxes)
        
        merged = []
        if dt_count > 0:
            # get centroid (x, y) of remaining bounding boxes
            centroid_x = (dt_bboxes[:, 0] + dt_bboxes[:, 2]) / 2
            centroid_y = (dt_bboxes[:, 1] + dt_bboxes[:, 3]) / 2
            if d > 0:
                for j in range(0, dt_count - 1):
                    if j not in merged:
                        for k in range(j + 1, dt_count):
                            diff_x = np.square(np.float32(centroid_x[j] - centroid_x[k]))
                            diff_y = np.square(np.float32(centroid_y[j] - centroid_y[k]))
                            distance = int(np.sqrt(diff_x + diff_y))
                            if distance <= d:
                                merged.append(k)
        # get new filtered detection count
        dt_count = int(dt_count - len(merged))
        image_info['count'].append(dt_count)

    df_image_info = pd.DataFrame(image_info)
    df_image_info.to_csv(path_counts_csv, index=False)
    print(TAG, f'\ncsv saved at {path_counts_csv}\n')

def calculate_kappa(gt_count, dt_count):
    kappa_gt_counts = [
        gt_count.loc[gt_count == 0].sum(),
        gt_count.loc[(gt_count >= 1) & (gt_count <= 5)].sum(),
        gt_count.loc[(gt_count >= 6) & (gt_count <= 10)].sum(),
        gt_count.loc[(gt_count >= 11) & (gt_count <= 20)].sum(),
        gt_count.loc[(gt_count >= 21) & (gt_count <= 50)].sum(),
        gt_count.loc[(gt_count >= 51) & (gt_count <= 200)].sum(),
        gt_count.loc[gt_count > 200].sum(),
    ]
    print('[calculate_kappa][kappa_gt_counts]', kappa_gt_counts)
    kappa_dt_counts = [
        dt_count.loc[dt_count == 0].sum(),
        dt_count.loc[(dt_count >= 1) & (dt_count <= 5)].sum(),
        dt_count.loc[(dt_count >= 6) & (dt_count <= 10)].sum(),
        dt_count.loc[(dt_count >= 11) & (dt_count <= 20)].sum(),
        dt_count.loc[(dt_count >= 21) & (dt_count <= 50)].sum(),
        dt_count.loc[(dt_count >= 51) & (dt_count <= 200)].sum(),
        dt_count.loc[dt_count > 200].sum(),
    ]
    print('[calculate_kappa][kappa_dt_counts]', kappa_dt_counts)
    kappa = metrics.quadratic_weighted_kappa(kappa_gt_counts, kappa_dt_counts)
    print('[calculate_kappa][kappa]', kappa)
    return kappa

def calculate_metrics(path_statistics_csv, title, path_metrics_csv):
    columns = ['distance', 'TP', 'FP', 'FN', 'recall', 'precision', 'fscore', 'accuracy', 'kappa']
    pr_curve_data = {key: [] for key in columns}
    df_image_info = pd.read_csv(path_statistics_csv)

    for d in range(20):
        condition = df_image_info.distance == d
        gt_count = df_image_info.loc[condition, 'gt_count']
        dt_count = df_image_info.loc[condition, 'dt_count']
        TP = df_image_info.loc[condition, 'TP'].sum()
        FP = df_image_info.loc[condition, 'FP'].sum()
        FN = df_image_info.loc[condition, 'FN'].sum()
        recall = TP / (TP + FN + 1e-10)
        precision = TP / (TP + FP) if not np.isnan(TP / (TP + FP)) else 1
        fscore = (2 * precision * recall) / (precision + recall)
        accuracy = TP / (TP + FP + FN)
        kappa = calculate_kappa(gt_count, dt_count)

        pr_curve_data['distance'].append(d)
        pr_curve_data['TP'].append(TP)
        pr_curve_data['FP'].append(FP)
        pr_curve_data['FN'].append(FN)
        pr_curve_data['recall'].append(recall)
        pr_curve_data['precision'].append(precision)
        pr_curve_data['fscore'].append(fscore)
        pr_curve_data['accuracy'].append(accuracy)
        pr_curve_data['kappa'].append(kappa)

    df_curve_data = pd.DataFrame(pr_curve_data)
    df_curve_data.to_csv(path_metrics_csv, index=False)

    return df_curve_data

# define global variables
TAG = '[z-final_model_script]'
path_configs = [
    './configs/lysto/fcn_unet_s5_s1_lysto.py', 
]

pr_curve_titles = [
    'FCN-UNet-ResNet50-s5-s1 | LYSTO',
]

ITER = 50000
for path_config, pr_curve_title in zip(path_configs, pr_curve_titles):
    cfg = Config.fromfile(path_config)
    cfg.load_from = os.path.join(cfg.work_dir, f'iter_{ITER}.pth')
    print(TAG, '[cfg.load_from]', cfg.load_from)
    # print(TAG, '[cfg.data.test.ann_file]', cfg.data.test.ann_file)
    cfg.resume_from = ''
    print(cfg.pretty_text)

    PATH_FINAL_MODEL_SCRIPT = os.path.join(cfg.PATH_WORK_DIR, 'final_model_script')
    if not os.path.exists(PATH_FINAL_MODEL_SCRIPT):
        os.mkdir(PATH_FINAL_MODEL_SCRIPT)
    
    # load dataset (1)
    # PATH_IMAGES1 = os.path.join(cfg.PATH_DATASET, 'test_DAB_images1')
    PATH_IMAGES1 = os.path.join('/home/zunaira/maskrcnn-lymphocyte-detection/mmdetection/lymphocyte_dataset/LYSTO-dataset/test_12000')
    PATH_IMAGES2 = os.path.join(cfg.PATH_DATASET, 'test_DAB_images2')
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

    # create model from config and load saved weights
    model = init_segmentor(cfg, cfg.load_from, device=device.type)
    # model.CLASSES = cfg.classes

    # calculate model's output for all images in the dataset and save it as numpy array
    path_model_output = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'outputs-12k-{cfg.MODEL_NAME}-s{cfg.S}-iter{ITER}.npz')
    if not os.path.exists(path_model_output):
        outputs = calculate_model_outputs(model, dataset, [PATH_IMAGES1], [LIST_DIR1], setting=cfg.S, path_model_output=path_model_output)
    else:
        print(TAG, '[path_model_output already exists]', path_model_output)
        outputs = np.load(path_model_output, allow_pickle=True)
        outputs = outputs['arr_0']
    print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)

    # calculate per-threshold, per-image statistics
    path_statistics_csv = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'stats-12k-{cfg.MODEL_NAME}-s{cfg.S}-iter{ITER}.csv')
    path_counts_csv = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'counts-12k-{cfg.MODEL_NAME}-s{cfg.S}-iter{ITER}.csv')
    if not os.path.exists(path_statistics_csv):
        # calculate_statistics_csv(cfg, outputs, dataset, path_statistics_csv, debug=False)
        calculate_counts_csv(outputs, dataset, path_counts_csv, debug=False)
    else:
        print(TAG, '[path_statistics_csv already exists]', path_statistics_csv)

    # # calculate and save PR-Curve using images-stats-per-threshold.csv
    # path_pr_curve = os.path.join(PATH_FINAL_MODEL_SCRIPT, f'pr-curve-12k-{cfg.MODEL_NAME}-s{cfg.S}-iter{ITER}.jpg')
    # df_curve_data = calculate_metrics(path_statistics_csv, pr_curve_title, path_pr_curve)
    # print(TAG, '[df_curve_data]\n', df_curve_data)
    # print(TAG, '[df_curve_data]\n', df_curve_data.describe())

    # path_outputs_drawn = os.path.join(cfg.work_dir, f'outputs-{cfg.MODEL_NAME}-s{cfg.S}-ep{EPOCH}')
    # if not os.path.exists(path_outputs_drawn):
    #     os.mkdir(path_outputs_drawn)
    #     draw_outputs_on_images(outputs, dataset, coco_annotations, PATH_IMAGES1, path_outputs_drawn)
    #     print(TAG, '[len(path_outputs_drawn)]', len(os.listdir(path_outputs_drawn)))
    # else:
    #     print(TAG, '[path_outputs_drawn already exists]', path_outputs_drawn)
