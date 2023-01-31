import os
import numpy as np
from tqdm import tqdm


def bbox_to_cxy(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array((cx, cy))


def get_filtered_preds(pred, dist_thrs=16):
    r = {}
    indices_to_remove = []
    for i in range(len(pred)):

        if i + 1 == len(pred):
            already_exists = False
            for k in list(r):
                if i in r[k]:
                    already_exists = True
            if already_exists:
                r[i] = [-1]
                indices_to_remove.append(i)
            else:
                r[i] = [i]

            break

        for j in range(i + 1, len(pred)):
            dist = np.linalg.norm(bbox_to_cxy(pred[i]) - bbox_to_cxy(pred[j]))
            if dist <= dist_thrs:
                if i in r.keys():
                    r[i].append(j)
                else:
                    already_exists = False
                    for k in list(r):
                        if i in r[k]:
                            already_exists = True
                            r[k].append(j)
                            r[i] = [-1]
                            indices_to_remove.append(i)
                    if not already_exists:
                        r[i] = [i, j]
                    else:
                        r[i] = [i]

            elif i not in list(r):
                already_exists = False
                for k in r.keys():
                    if i in r[k]:
                        already_exists = True
                if already_exists:
                    r[i] = [-1]
                    indices_to_remove.append(i)
                else:
                    r[i] = [i]

    filtered_indexes = list(set(r.keys()) - set(indices_to_remove))
    filtered_preds = [pred[i] for i in filtered_indexes]

    return filtered_indexes, filtered_preds


def code2(bboxes, d):
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
    filtered_indexes = list(set(range(dt_count)) - set(merged))
    filtered_preds = [bboxes[i] for i in filtered_indexes]

    return filtered_indexes, filtered_preds


def verify_distance_codes(outputs, dataset):
    threshold = 0.5
    for d in [0, 12, 16]:
        print(TAG, f'running loop for d = {d}')
        filtered_indexes1_count = 0
        filtered_indexes2_count = 0
        for id, image_name in enumerate(dataset):
            if id > len(outputs): break
            output = outputs[id]
            scores = output[0][0][:, 4]
            bboxes = output[0][0][:, :4]
            bboxes = bboxes[np.where(scores >= threshold)[0]]

            filtered_indexes1, filtered_preds1 = get_filtered_preds(bboxes, d)
            filtered_indexes2, filtered_preds2 = code2(bboxes, d)
            filtered_indexes1_count += len(filtered_indexes1)
            filtered_indexes2_count += len(filtered_indexes2)

        print(f'[filtered_indexes1_count]', filtered_indexes1_count)
        print(f'[filtered_indexes2_count]', filtered_indexes2_count)
        print()

TAG = '[z-verify_distance_code_01]'
PATH_DATASET = 'lymphocyte_dataset/LYSTO-dataset'
base_path_dataset = os.path.join(PATH_DATASET, 'test_12000')
print(TAG, '[base_path_dataset]', base_path_dataset)
dataset = os.listdir(base_path_dataset)
print(TAG, '[len(dataset)]', len(dataset), dataset[:10], dataset[-10:])
dataset = sorted(dataset, key=lambda x: int(x[:-4].split('_')[1]))
print(TAG, '[len(dataset)]', len(dataset), dataset[:10], dataset[-10:])

PATH_INFERENCE_LYSTO_12000 = 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting13/combined/inference_lysto_12000'
path_model_output = os.path.join(PATH_INFERENCE_LYSTO_12000, f'outputs-12k-maskrcnn-lymphocytenet3-cm1-s13.npz')
print(TAG, '[path_model_output]', path_model_output)
outputs = np.load(path_model_output, allow_pickle=True)
outputs = outputs['arr_0']
print(TAG, '[type(outputs)]', type(outputs), '[outputs.dtype]', outputs.dtype, '[outputs.shape]', outputs.shape)

verify_distance_codes(outputs, dataset)

# [z-verify_distance_code_01] running loop for d = 0
# [filtered_indexes1_count] 41644
# [filtered_indexes2_count] 41648

# [z-verify_distance_code_01] running loop for d = 12
# [filtered_indexes1_count] 40300
# [filtered_indexes2_count] 40296

# [z-verify_distance_code_01] running loop for d = 16
# [filtered_indexes1_count] 40278
# [filtered_indexes2_count] 40267
