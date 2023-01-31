import os
import cv2
import csv
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2hed, hed2rgb

import torch
import torch.nn.functional as F

import classification.clf_models as mdl
import pretrainedmodels as ptm
import classification.helpers as utils
from hybrid.configs import Configurations
from detectron2.engine import DefaultPredictor

# 0. Preliminaries and Constants
imgs_root = '/home/mohsin/Projects/lysto_datasets/resized_cropped/'
fnames = [str(i) + '.png' for i in range(1, 12001)]

id2cls = {
    0: 'Artifact',
    1: 'Normal',
    2: 'Lymphocyte'
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# empty cuda cache
torch.cuda.empty_cache()


# dist_thrs = 14
def bbox_to_cxy(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array((cx, cy))


def get_filtered_preds(pred, dist_thrs=8):  # dist_thr changed to 8 from 13
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


# 1. Load Classification Model
# model_obj = mdl.ModifiedInceptionV3(num_classes=3, use_custom_block=True)
# model_obj = mdl.ModifiedResNet101(num_classes=3, use_custom_block=False)
# model = model_obj.get_model()
# model_name = model_obj.model_name
# model = mdl.PCNet3P2(in_channels=3, num_classes=3)
# model_name = model.model_name
model_name = 'se_resnet101'
model = ptm.__dict__[model_name](num_classes=3, pretrained=None)

if model_name == 'inception3':
    img_size = (299, 299)
    for_model = 'inception3'
elif model_name == 'resnet50' or model_name == 'resnet101' or model_name == 'se_resnet101':
    img_size = (224, 224)
    for_model = 'resnet'
else:
    img_size = (256, 256)
    for_model = 'custom'

transformations = utils.get_transformations(for_train=False, resize=img_size, for_model=for_model)
stage_1_images = os.path.join(imgs_root, 'test_org')
cp_path = os.path.join(os.getcwd(), 'checkpoints', 'se_resnet101', 'model_multi_01_se_resnet101_best_weights.pth')
model.to(device=device)
model.load_state_dict(torch.load(cp_path))
model.eval()

# 2. Load Detection Model
detection_model = 'mrcnn_mixed_dab'
cf = Configurations('mrcnn', detection_model)
cfg = cf.get_configurations()
predictor = DefaultPredictor(cfg)
stage_2_images = os.path.join(imgs_root, 'test_dab')

# 3. Construct Hybrid Pipeline
#   Original RGB Input
#       |-> Classification Model
#           |-> A tensor 1x3
#               |-> Label Based on Argmax and Scores using Softmax
#                   |-> If not Artifact then pass
#                       |-> Detection Model

result_analysis = []

for f in tqdm(fnames):

    input_alpha_fp = os.path.join(stage_1_images, f)
    input_alpha = Image.open(input_alpha_fp)
    input_alpha = transformations(input_alpha).unsqueeze(0)

    output_alpha = model(input_alpha.to(device))
    _, predicted = torch.max(output_alpha.data, 1)
    classification_scores = F.softmax(output_alpha).data.cpu().numpy().ravel().tolist()
    # classification_scores = [round(x, 3) for x in classification_scores]
    predicted_label = predicted.item()
    predicted_class = id2cls[predicted_label]

    # ---------------------- Going towards stage 2 ----------------------------
    TWO_STAGE = True
    if TWO_STAGE:
        if predicted_label == 0:
            predicted_count = 0
        else:
            input_beta_fp = os.path.join(stage_2_images, f)
            input_beta = cv2.imread(input_beta_fp)
            output_beta = predictor(input_beta)
            predicted_boxes = output_beta['instances'].pred_boxes.to('cpu').tensor.tolist()
            idx, filtered_boxes = get_filtered_preds(predicted_boxes)
            predicted_count = len(filtered_boxes)
            # predicted_count = len(predicted_boxes)
    else:
        input_beta_fp = os.path.join(stage_2_images, f)
        input_beta = cv2.imread(input_beta_fp)
        output_beta = predictor(input_beta)
        predicted_boxes = output_beta['instances'].pred_boxes.to('cpu').tensor.tolist()
        idx, filtered_boxes = get_filtered_preds(predicted_boxes)
        predicted_count = len(filtered_boxes)

    result = [f.split('.')[0], predicted_count, predicted_class] + classification_scores
    result_analysis.append(result)

test_results_analysis_df = pd.DataFrame(data=result_analysis,
                                        columns=['id', 'count', 'predicted_class',
                                                 'score_artifact', 'score_normal', 'score_lymphocyte'])
test_results_analysis_df.to_csv(os.path.join('results', 'test_release', 'test_' + model_name + '_' + detection_model + 'thr_048_fil_8' + '.csv'),
                                index=False)
# test_results_analysis_df.to_csv(os.path.join('results', 'test_release', 'test_' + 'single_stage' + '_' + detection_model + '.csv'),
#                                 index=False)
submission_df = test_results_analysis_df[['id', 'count']]
submission_df.to_csv(os.path.join('results', 'test_release', 'submission_09.csv'), index=False)
