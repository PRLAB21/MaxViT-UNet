import sys

sys.path.append('/home/zunaira/project')

import os
import cv2
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from skimage.color import rgb2hed, hed2rgb

import torch
import torch.nn.functional as F

# import classification.clf_models as mdl
# import pretrainedmodels as ptm
import classification.helpers as utils
import classification.lymp_net2_3class as c_model
from hybrid.configs import Configurations

from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed, inference_detector, init_detector
from mmdet.core.utils import ml_metrics as metrics

from mmdet.utils.lysto_utils_pipeline import *

set_random_seed(0, deterministic=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# empty cuda cache
torch.cuda.empty_cache()
print(f'running on {device}')

# 0. Preliminaries and Constants
# imgs_root = r'/home/zunaira/lyon_dataset/lyon_patch_overlap_onboundries'
imgs_root = r'/home/zunaira/lyon_dataset/lyon_dataset_small/Validation'
# lysto_img_dir = r''
fnames = os.listdir(imgs_root)
print('[fnames]', fnames[:10], fnames[-10:])
print('[fnames]', len(fnames))
fnames = sorted(fnames, key=lambda x: (int(x[4:-4].split('-')[0]), int(x[4:-4].split('-')[1])))
print('[fnames]', fnames[:10], fnames[-10:])
print('[fnames]', len(fnames))

# 2. Load Detection Model
# detection_model = 'LymphNet_newArch' #'lyon_LymphNet'
# cf = Configurations('LymphNet_newArch', detection_model) #'lyon_LymphNet'
# cfg = cf.get_configurations()
# predictor = DefaultPredictor(cfg)
# #stage_2_images = os.path.join(imgs_root, 'test_dab')

# ******************* LOAD MMDETECTION MODEL *******************
# path_config = 'configs/lymphocyte/maskrcnn_lymphocytenet3_cm1_s13_lysto_combined.py'
path_config = 'configs/lyon/maskrcnn_lymphocytenet3_cm1_s14_lyon.py'
cfg = Config.fromfile(path_config)
cfg.load_from = os.path.join(cfg.work_dir, f'latest.pth')
print('[cfg.load_from]', cfg.load_from)
cfg.resume_from = ''
# print(cfg.pretty_text)
# create detection_model from config and load the trained weights
detection_model = init_detector(cfg, cfg.load_from, device=device.type)
detection_model.CLASSES = cfg.classes
detection_model.cfg = cfg
test_df = pd.read_csv('lyon_test_labels.csv')
# print('[detection_model]\n', detection_model)
# **************************************************************

# 3. Construct Hybrid Pipeline
#   Original RGB Input
#       |-> Classification Model
#           |-> A tensor 1x3
#               |-> Label Based on Argmax and Scores using Softmax
#                   |-> If not Artifact then pass
#                       |-> Detection Model

result_analysis = []

# fnames = fnames[:1500]
for i, f in tqdm(enumerate(fnames), total=len(fnames), position=0, leave=True):
    # load image for classification model
    input_alpha_fp = os.path.join(imgs_root, f)
    input_alpha = Image.open(input_alpha_fp)
    # apply normalization
    # input_alpha = normalize_255(input_alpha)
    # apply transformations for classification model
    input_alpha = cls_transformations(input_alpha).unsqueeze(0)
    # get inference from classification model
    output_alpha = cls_model(input_alpha.to(device))
    # extract predicted class and confidence score
    _, predicted = torch.max(output_alpha.data, 1)
    classification_scores = F.softmax(output_alpha, dim=1).data.cpu().numpy().ravel().tolist()
    # classification_scores = [round(x, 3) for x in classification_scores]
    predicted_label = predicted.item()
    predicted_class = id2cls[predicted_label]

    # ---------------------- Going towards stage 2 ----------------------------
    TWO_STAGE = True
    if TWO_STAGE:
        if predicted_label == 0 or predicted_label == 1:  # filter artifacts and normal patches
            predicted_count = 0
            # result = [f, -1, -1, predicted_class]
            # result_analysis.append(result)
            # centerCoord[0] = 0
            # centerCoord[1] = 0
        else:
            # load image for detection model
            input_beta_fp = os.path.join(imgs_root, f)
            input_beta = mmcv.imread(input_beta_fp)
            # normalize RGB image
            input_beta = cv2.cvtColor(input_beta, cv2.COLOR_BGR2RGB)
            input_beta = normalize_255(input_beta)
            # input_beta = ihc2dab(input_beta) # RGB-DAB
            input_beta = cv2.cvtColor(input_beta, cv2.COLOR_RGB2BGR)
            # get inference from detection model
            output_beta = inference_detector(detection_model, input_beta)

            # v = Visualizer(input_beta[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            # v = v.draw_instance_predictions(output_beta["instances"].to("cpu"))
            # ipath = r'/home/zunaira/project/hybrid/results/vis_output/newEval'
            # impath = os.path.join(ipath, f)
            # plt.imsave(impath, cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))

            # extract bounding box predictions
            original_count = test_df.loc[test_df ['image_id'] == f].n_lymphs.values[0]
            predicted_boxes = output_beta[0][0]
            predicted_boxes = predicted_boxes[predicted_boxes[:, 4] >= 0.3]
            idx, filtered_boxes = get_filtered_preds(predicted_boxes.tolist())
            filtered_count = len(filtered_boxes)
            predicted_count = len(predicted_boxes)
            # for rect in filtered_boxes:
            #     centerCoord = bbox_to_cxy(rect)
            #     result = [f, centerCoord[0], centerCoord[1], predicted_class]
            #     result_analysis.append(result)
                # predicted_count = len(predicted_boxes)
            tp = 0
            fp = 0
            fn = 0
            if original_count == filtered_count:
                tp = original_count
                # tp = 1
            elif original_count < filtered_count:
                tp = original_count
                fp = abs(original_count - filtered_count)
                # fp = 1
            elif original_count > filtered_count:
                tp = filtered_count
                fn = abs(original_count - filtered_count)
                # fn = 1
            result_analysis.append([f, original_count, filtered_count, tp, fp, fn])
    else:
        input_beta_fp = os.path.join(imgs_root, f)
        input_beta = mmcv.imread(input_beta_fp)
        input_beta = cv2.cvtColor(input_beta, cv2.COLOR_BGR2RGB)
        # input_beta = ihc2dab(input_beta)
        input_beta = cv2.cvtColor(input_beta, cv2.COLOR_RGB2BGR)
        output_beta = inference_detector(detection_model, input_beta)

        predicted_boxes = output_beta[0][0]
        predicted_boxes = predicted_boxes[predicted_boxes[:, 4] >= 0.3]
        idx, filtered_boxes = get_filtered_preds(predicted_boxes.tolist())
        filtered_count = len(filtered_boxes)
        predicted_count = len(predicted_boxes)
        for rect in filtered_boxes:
            centerCoord = bbox_to_cxy(rect)
            result = [f, predicted_count, filtered_count, centerCoord[0], centerCoord[1], predicted_class] + classification_scores
            result_analysis.append(result)
    # break

print('[len(result_analysis)]', len(result_analysis))
# test_results_analysis_df = pd.DataFrame(data=result_analysis, columns=['image_id', 'x', 'y', 'class_label'], encoding= "utf_8")

test_results_analysis_df = pd.DataFrame(data=result_analysis, columns=['x', 'y', 'y_pred', 'tp', 'fp', 'fn'])
# lyon_test_results_df.to_csv(os.path.join(r'/home/zunaira/project/hybrid/results/lyon/internaltest_results', 'test_analysis_lyon.csv'), index=False)
cum_tp = test_results_analysis_df['tp'].sum()
cum_fp = test_results_analysis_df['fp'].sum()
cum_fn = test_results_analysis_df['fn'].sum()
precision = cum_tp / (cum_tp + cum_fp)
recall = cum_tp / (cum_tp + cum_fn)
f1_score = (2 * precision * recall) / (precision + recall)

print('[test_results_analysis_df]\n', test_results_analysis_df)
print('Precision: {0}\nRecall: {1}\nF1 Score: {2}'.format(precision, recall, f1_score))
resultf = [[f1_score, precision, recall]]
result_file_df = pd.DataFrame(data=resultf, columns=['F', 'P', 'R'])

inference_pipeline_path = os.path.join(cfg.work_dir, 'inference_pipeline_lyon_internal')
if not os.path.exists(inference_pipeline_path):
    os.mkdir(inference_pipeline_path)
df_path = os.path.join(inference_pipeline_path, f'{model_name}-{cfg.MODEL_NAME}-s{cfg.S}Filt_T0.3.csv')
test_results_analysis_df.to_csv(df_path, index=False)
df2_path = os.path.join(inference_pipeline_path, f'{model_name}-{cfg.MODEL_NAME}-s{cfg.S}-Scores_Filt_T0.3.csv')
result_file_df.to_csv(df2_path, index=False)
# submit_test_df = test_results_analysis_df[['image_id', 'x', 'y']]
# test_results_analysis_df.to_csv(os.path.join('results', 'test_release', 'test_' + 'single_stage' + '_' + detection_model + '.csv'), index=False)
# submission_df = test_results_analysis_df[['image_id', 'x', 'y']]
# submission_df.to_csv(os.path.join(cfg.work_dir, 'z-inference_lysto-e30-submit_newclasswithnewdet.csv'), index=False)
print('[end]')
