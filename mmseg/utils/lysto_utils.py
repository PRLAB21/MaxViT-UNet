import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm

import mmcv
from mmseg.apis import inference_segmentor
from mmseg.core.utils import ml_metrics

TAG = '[lysto_utils]'

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
    return ihc_d

def ihc2hsv(ihc_rgb):
    # img = cv2.imread(ihc_rgb)
    img = cv2.cvtColor(ihc_rgb, cv2.COLOR_BGR2RGB)
    # Convert the RGB image to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img_hsv

def calculate_model_outputs(model, dataset, base_path_images, path_model_output, is_dab=False):
    TAG2 = TAG + '[calculate_model_outputs]'
    # get model's prediction on all images
    outputs = []
    for image_idx, image_name in enumerate(mmcv.track_iter_progress(dataset)):
        # create image path
        img_path = os.path.join(base_path_images, image_name)
        # load image
        img_original = mmcv.imread(img_path)
        if is_dab:
            img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img_original = ihc2dab(img_original)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        # get inference
        output = inference_segmentor(model, img_original)
        # append this output
        outputs.append(output)
        # if image_idx > 19: break
        if (image_idx + 1) % 10000 == 0:
            # save the model's predictions as numpy compressed array
            np.savez_compressed(path_model_output[:-4] + str(image_idx) + '.npz', np.asarray(outputs))
    
    # convert python list to numpy array
    outputs = np.asarray(outputs, dtype=object)
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

def get_contours_from_mask(mask_image_path=None, mask_image=None):
    TAG2 = TAG + '[get_contours_from_mask]'
    if mask_image_path is not None:
        # print('[mask_image_path]', mask_image_path)
        im = cv2.imread(mask_image_path)
        imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if len(im.shape) > 2 else im
    elif mask_image is not None:
        imgray = mask_image
    # print('[imgray]', imgray.dtype, imgray.shape, imgray.min(), imgray.max())
    if len(np.unique(imgray)) > 1:
        # print(np.unique(imgray))
        imgray = normalize_255(imgray)
    else:
        imgray = imgray.astype(np.uint8)
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

def draw_outputs_with_original(outputs, dataset, base_path_images, mask_images_dir, path_outputs_drawn):
    print(TAG, '[draw_outputs_with_original][1]')
    distance = 12
    font_size = 0.5
    font_weight = 1
    resize_scale = 2

    for idx, (image_name, dt_mask) in tqdm(enumerate(zip(dataset, outputs)), total=len(dataset)):
        # get gt_bboxes
        gt_mask_path = os.path.join(mask_images_dir, image_name)

        gt_contours, gt_count = get_contours_from_mask(mask_image_path=gt_mask_path)
        dt_contours, dt_count1 = get_contours_from_mask(mask_image=dt_mask[0])

        gt_bboxes = get_bboxes_from_contours(gt_contours)
        dt_bboxes = get_bboxes_from_contours(dt_contours)
        gt_bboxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in gt_bboxes])
        dt_bboxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in dt_bboxes])

        dt_count1 = int(len(dt_bboxes))
        if dt_count1 > 0:
            centroid_x = dt_bboxes[:, 0] + dt_bboxes[:, 2] / 2
            centroid_y = dt_bboxes[:, 1] + dt_bboxes[:, 3] / 2
            merged = []
            for j in range(0, dt_count1 - 1):
                if j not in merged:
                    for k in range(j + 1, dt_count1):
                        diff_x = np.float32(np.square(centroid_x[j] - centroid_x[k]))
                        diff_y = np.float32(np.square(centroid_y[j] - centroid_y[k]))
                        distance = int(np.sqrt(diff_x + diff_y))
                        if distance <= 12:
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
                        # score_text = f"FP({int(100 * bbox[4])}%)"
                        score_text = f"FP"
                        bg_color = (0, 0, 255)
                    # true positive
                    elif sum(iou_matrix[:, j]) > 0:
                        # score_text = f"TP({int(100 * bbox[4])}%)"
                        score_text = f"TP"
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

def draw_outputs_without_original(outputs, dataset, base_path_images, path_outputs_drawn):
    TAG2 = TAG + '[draw_outputs_without_original]'
    print(TAG2, '[starts]')
    d = 12
    threshold = 0.8
    font_size = 0.5
    font_weight = 1
    resize_scale = 2

    for idx, (image_name, output) in tqdm(enumerate(zip(dataset, outputs)), total=len(dataset)):
        if idx > 1999: break
        # get dt_bboxes
        dt_bboxes = output[0][0]
        scores = dt_bboxes[:, 4]
        dt_masks = output[1][0]
        # print(TAG2, idx, '[dt_bboxes, dt_masks]', dt_bboxes.shape, dt_masks.shape)
        # dt_bboxes = dt_bboxes[:, :]
        dt_bboxes = dt_bboxes[np.where(scores >= threshold)[0]]
        dt_masks = dt_masks[np.where(scores >= threshold)[0]]
        # print(TAG2, idx, '[dt_bboxes, dt_masks]', dt_bboxes.shape, dt_masks.shape)
        dt_count1 = len(dt_bboxes)
        centroid_x = (dt_bboxes[:, 0] + dt_bboxes[:, 2]) / 2
        centroid_y = (dt_bboxes[:, 1] + dt_bboxes[:, 3]) / 2
        merged = []
        for j in range(0, dt_count1 - 1):
            if j not in merged:
                for k in range(j + 1, dt_count1):
                    diff_x = np.square(np.float32(centroid_x[j] - centroid_x[k]))
                    diff_y = np.square(np.float32(centroid_y[j] - centroid_y[k]))
                    distance = int(np.sqrt(diff_x + diff_y))
                    if distance <= d:
                        merged.append(k)
        dt_bboxes = np.delete(dt_bboxes, merged, axis=0)
        dt_masks = np.delete(dt_masks, merged, axis=0)
        # print(TAG2, idx, '[dt_bboxes, dt_masks]', dt_bboxes.shape, dt_masks.shape)

        # load image
        img_path = os.path.join(base_path_images, image_name)
        image = cv2.imread(img_path)
        # print(TAG2, idx, '[image]', image.shape)
        image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
        # print(TAG2, idx, '[image]', image.shape)

        for j, bbox in enumerate(dt_bboxes):
            score_text = f"Lymphocyte ({int(100 * bbox[4])}%)"
            bg_color = (0, 128, 0)
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
            x2 = x1 + 75
            y2 = y1 - 25
            if x2 >= image.shape[1]:
                x2 = int(bbox[2]) * resize_scale
                x1 = x2 - 75
            if y2 <= 0:
                y2 = int(bbox[3]) * resize_scale
                y1 = y2 + 25
            # draw rectangle for text bg
            cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
            # draw text
            cv2.putText(image, score_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_weight, cv2.LINE_AA)

            mask = dt_masks[j]
            # print(TAG2, idx, '[mask.shape]', mask.shape)
            mask = np.dstack([mask, mask, mask]).astype(np.uint8)
            mask = cv2.resize(mask, image.shape[:2])
            mask = mask[:, :, 0]
            indices = np.where(mask == 1)
            # print(TAG2, idx, '[mask.shape]', mask.shape)
            # print(TAG2, idx, len(indices))
            overlay = np.zeros_like(image)
            overlay[indices[0], indices[1], 0] = 255
            dt_contours, dt_count = get_contours_from_mask(mask_image=mask)
            # print(TAG2, idx, '[dt_contours]', len(dt_contours), dt_contours[0].dtype)
            image[indices[0], indices[1], :] = cv2.addWeighted(image[indices[0], indices[1], :], 0.5, overlay[indices[0], indices[1], :], 0.5, gamma=1)
            cv2.drawContours(image, dt_contours, -1, (0, 0, 255), 5, cv2.LINE_AA)

        filename = os.path.join(path_outputs_drawn, image_name)
        cv2.imwrite(filename, image)

def calculate_centroids_csv(outputs, dataset, path_centroids_csv, labels_csv=None, debug=False):
    columns = ['id', 'image_name', 'center_x', 'center_y', 'x1', 'y1', 'x2', 'y2']
    if labels_csv is not None:
        columns += ['organ', 'organ_type']
    print(TAG, '[columns]', columns)
    centroids_info = {key: [] for key in columns}
    
    # iterate over all dataset images
    for id, image_name in enumerate(mmcv.track_iter_progress(dataset)):
        image_name = image_name[:-4] + '.png'
        row = labels_csv[labels_csv.x == image_name] if labels_csv is not None else None

        # get model's prediction
        dt_mask = outputs[id][0]
        dt_contours, dt_count1 = get_contours_from_mask(mask_image=dt_mask)
        dt_bboxes = get_bboxes_from_contours(dt_contours)

        # append data of filtered bboxes in dictionary
        for box in dt_bboxes:
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

def calculate_statistics_csv(outputs, dataset, gt_mask_dir, path_statistics_csv, debug=False):
    TAG2 = TAG + '[calculate_statistics_csv]'
    columns = ['distance', 'image_name', 'gt_count', 'dt_count_d0', 'dt_count_d12']
    image_info = {key: [] for key in columns}
    # print('[coco_annotations]', coco_annotations)

    # iterate over all dataset images
    for d in [12]:
        print(TAG, f'running loop for d = {d}')
        for id, image_name in enumerate(tqdm(dataset, total=len(dataset))):
            # store info
            image_info['distance'].append(d)
            image_info['image_name'].append(image_name)
            
            gt_mask_path = os.path.join(gt_mask_dir, image_name)
            dt_mask = outputs[id][0]
            # print(TAG2, '[dt_mask]', type(dt_mask), dt_mask.dtype, np.unique(dt_mask))
            
            gt_contours, gt_count = get_contours_from_mask(mask_image_path=gt_mask_path)
            dt_contours, dt_count1 = get_contours_from_mask(mask_image=dt_mask)

            gt_bboxes = get_bboxes_from_contours(gt_contours)
            dt_bboxes = get_bboxes_from_contours(dt_contours)
            gt_count = len(gt_bboxes)
            dt_count1 = len(dt_bboxes)

            merged = []
            if dt_count1 > 0:
                # get centroid (x, y) of remaining bounding boxes
                centroid_x = (dt_bboxes[:, 0] + dt_bboxes[:, 2]) / 2
                centroid_y = (dt_bboxes[:, 1] + dt_bboxes[:, 3]) / 2
                if d > 0:
                    for j in range(0, dt_count1 - 1):
                        if j not in merged:
                            for k in range(j + 1, dt_count1):
                                diff_x = np.square(np.float32(centroid_x[j] - centroid_x[k]))
                                diff_y = np.square(np.float32(centroid_y[j] - centroid_y[k]))
                                distance = int(np.sqrt(diff_x + diff_y))
                                if distance <= d:
                                    merged.append(k)
            # else:
            #     print(TAG2, '[dt_bboxes]', len(dt_bboxes), image_name)
            
            # get new filtered detection count
            dt_count2 = int(dt_count1 - len(merged))
            image_info['gt_count'].append(gt_count)
            image_info['dt_count_d0'].append(dt_count1)
            image_info[f'dt_count_d{d}'].append(dt_count2)

        df_image_info = pd.DataFrame(image_info)
        df_image_info.to_csv(path_statistics_csv, index=False)
        print(TAG, f'\ncsv saved at {path_statistics_csv}\n')

def calculate_counts_csv(outputs, dataset, path_statistics_csv, d=12, threshold=0.5, is_segmentation=True, debug=False):
    columns = ['id', 'count']
    image_info = {key: [] for key in columns}
    threshold = np.round(threshold, 2)

    # iterate over all dataset images
    print(TAG, f'd={d}, threshold={threshold}')
    for id, image_name in tqdm(enumerate(dataset), total=len(dataset)):
        # store info
        image_info['id'].append(int(image_name[5:-4]))
        # get model prediction
        output = outputs[id]
        # classification scores
        scores = output[0][0][:, 4] if is_segmentation else output[0][:, 4]
        # bounding boxes
        bboxes = output[0][0][:, :4] if is_segmentation else output[0][:, :4]
        # filter boxes with score more then threshold
        bboxes = bboxes[np.where(scores >= threshold)[0]]
        # store detected count
        dt_count = int(len(bboxes))
        # get centroid (x, y) of remaining bounding boxes
        centroid_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        centroid_y = (bboxes[:, 1] + bboxes[:, 3]) / 2

        merged = []
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
    # path_statistics_csv = path_statistics_csv[:-4] + f'-d_{d}-t_{int(threshold*100)}.csv'
    df_image_info.to_csv(path_statistics_csv, index=False)
    print(TAG, f'\ncsv saved at {path_statistics_csv}\n')

# def calculate_pr_curve(path_statistics_csv, thresholds, title, path_pr_curve):
#     df_image_info = pd.read_csv(path_statistics_csv)
#     TPs, FPs, FNs = [], [], []
#     recalls, precisions, fscores = [], [], []
#     for threshold in thresholds:
#         threshold = np.round(threshold, 2)
#         condition = (df_image_info.distance == 12) & (df_image_info.threshold == threshold)
#         TP = df_image_info.loc[condition, 'TP'].sum()
#         FP = df_image_info.loc[condition, 'FP'].sum()
#         FN = df_image_info.loc[condition, 'FN'].sum()
#         # TN = df_image_info.loc[condition, 'TN'].sum()
#         TPs.append(TP), FPs.append(FP), FNs.append(FN)
#         recall = TP / (TP + FN + 1e-10)
#         precision = TP / (TP + FP)
#         precision = 1 if np.isnan(precision) else precision
#         fscore = (2 * precision * recall) / (precision + recall)
#         recalls.append(recall)
#         precisions.append(precision)
#         fscores.append(fscore)

#     idx_max_fscore = np.array(fscores).argmax()
#     print(TAG, f'Max F-Score = {fscores[idx_max_fscore]} | at index = {idx_max_fscore}')

#     df = pd.DataFrame({'thresholds': thresholds, 'TPs': TPs, 'FPs': FPs, 'FNs': FNs, 'recalls': recalls, 'precisions': precisions, 'fscores': fscores})

#     plt.figure(figsize=(10, 10))
#     plt.suptitle(f'PR-Curve | {title}', fontsize=20)
#     plt.plot(recalls, precisions)
#     # plt.plot(recalls, precisions, 'o')
#     plt.xlabel('Recall', fontsize=20)
#     plt.ylabel('Precision', fontsize=20)
#     plt.xlim((0, 1))
#     plt.ylim((0, 1))
#     # plt.plot((recalls[idx_max_fscore], recalls[idx_max_fscore]), (0, 1))
#     # plt.plot((0, 1), (precisions[idx_max_fscore], precisions[idx_max_fscore]))
#     plt.savefig(path_pr_curve, dpi=150)
#     print(TAG, f'plot saved at {path_pr_curve}')
    
#     return df
