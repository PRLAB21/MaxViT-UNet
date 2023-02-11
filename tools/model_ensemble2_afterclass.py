# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from skimage.measure import label, regionprops

import torch

import mmcv
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmcls.apis import init_model, inference_model

from mmdet.apis import inference_detector
from mmdet.models import build_detector

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.apis import inference_segmentor
from mmseg.models import build_segmentor
from mmseg.utils import lysto_utils
from mmseg.core.utils import ml_metrics as metrics

TAG = f"{os.path.basename(__file__).split('.')[0]}"
osp = os.path

def normalize_255(image):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    return image

def bbox_to_cxy(bbox):
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return np.array((cx, cy))

def load_models(configs, checkpoints, gpu_ids):
    TAG2 = TAG + '[load_models]'
    models = []
    
    for idx, (config, ckpt) in enumerate(zip(configs, checkpoints)):
        print(TAG2, '[config]', config)
        model_type = 'mmseg' if 'mmsegmentation' in config else 'mmdet'
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        
        model = build_segmentor(cfg.model) if model_type == 'mmseg' else build_detector(cfg.model)
        # print(TAG2, '[model]')
        # print(model)
        # model = init_segmentor(cfg, ckpt) if model_type == 'mmseg' else 
        if cfg.get('fp16', None):
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()

        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        model.cfg = cfg
        # appending all the segmentation models
        models.append((model_type, model))
    
    return models

@torch.no_grad()
def main(args):
    TAG2 = TAG + '[main]'
    imgs_root = args.testpath
    # r'/home/gpu02/lyon_dataset/lyon_patch_overlap_onboundries'
    # lysto_img_dir = r''
    models = []
    gpu_ids = args.gpus
    configs = args.config
    checkpoints = args.checkpoint
    class_config = args.class_confg
    class_ckpt = args.class_checkpoint
    model_name = args.model_name

    cfg = mmcv.Config.fromfile(configs[0])

    # if args.aug_test:
    #     cfg.data.test.pipeline[1].img_ratios = [
    #         0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
    #     ]
    #     cfg.data.test.pipeline[1].flip = True
    # else:
    #     cfg.data.test.pipeline[1].img_ratios = [1.0]
    #     cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True
    # output dir path
    tmpdir = args.out
    mmcv.mkdir_or_exist(tmpdir)
    img_path = osp.join(tmpdir, 'imgs')
    mmcv.mkdir_or_exist(img_path)
    # file names of test data
    print(TAG2, '[imgs_root]', imgs_root)
    fnames = os.listdir(imgs_root)
    print(TAG2, '[fnames]', fnames[:10], fnames[-10:])
    print(TAG2, '[fnames]', len(fnames))
    fnames = sorted(fnames, key=lambda x: (int(x[4:-4].split('-')[0]), int(x[4:-4].split('-')[1])))
    print(TAG2, '[fnames]', fnames[:10], fnames[-10:])
    print(TAG2, '[fnames]', len(fnames))
    outputs = [] # to save the combined results
    open_disk_r = 10

    # CLASSIFICATION FOLLOWED BY SEGMENTATION
    if class_config is not None:

        # 1. Load Classification Model
        ####### load stage 1 model (LYMPHOCYTE vs NO-LYMPHOCYTE) #######
        # cfg_path1 = '/home/gpu02/maskrcnn-lymphocyte-detection/mmclassification/configs/lysto_lymph_vs_nolymph/lympnet2_s2.py'
        cfg1 = mmcv.Config.fromfile(class_config)
        # print(TAG, '[cfg1]\n', cfg1.pretty_text)
        # print(TAG2, 'loading model_stage1 from class_ckpt:', class_ckpt)
        model_stage1 = init_model(cfg1, class_ckpt)
        model_stage1.CLASSES = cfg1.classes
        print(TAG2, '[model_stage1.classes]', model_stage1.CLASSES)

        # 2. Load Segmentation Models (Ensembled)
        models = load_models(configs, checkpoints, gpu_ids)

        prog_bar = mmcv.ProgressBar(len(fnames))

        result_analysis = []
        df_path = osp.join(tmpdir, f'{model_name}_majority.csv')
        print(TAG2, '[df_path]', df_path)

        # images_done = set(pd.read_csv(df_path)['image_id'].to_list())
        # print('[images_done]', len(images_done))
        # images_remaining = set(fnames).difference(images_done)
        # df_path = opj(inference_pipeline_path, f'baseline_unet-02.csv')
        # fnames = sorted(list(images_remaining), key=lambda x: (int(x[4:-4].split('-')[0]), int(x[4:-4].split('-')[1])))
        # print('[fnames]', fnames[:10], fnames[-10:])
        # print('[fnames]', len(fnames))
        # exit()

        # *****inference****
        for i, image_name in tqdm(enumerate(fnames), total=len(fnames), position=0, leave=True):

            ####### MMClassification Model #######
            # load image
            image_path = osp.join(imgs_root, image_name)
            image = cv2.imread(image_path)

            # forward pass
            output = inference_model(model_stage1, image)
            predicted_label = output['pred_label']
            preds = 0

            if predicted_label == 0: # or predicted_label == 1:  # filter artifacts and normal patches
                predicted_count = 0
            else:
                # load image for detection model
                for model_type, model in models:
                    input_beta_fp = osp.join(imgs_root, image_name)
                    input_beta = mmcv.imread(input_beta_fp)
                    # normalize RGB image
                    input_beta = cv2.cvtColor(input_beta, cv2.COLOR_BGR2RGB)
                    input_beta = normalize_255(input_beta)
                    # input_beta = ihc2dab(input_beta) # RGB-DAB
                    input_beta = cv2.cvtColor(input_beta, cv2.COLOR_RGB2BGR)
                    if model_type == 'mmseg':
                        result = inference_segmentor(model, input_beta)
                        result = result[0]
                    else:
                        result = inference_detector(model, input_beta)
                        result[1][0] = np.stack(result[1][0], axis=0) if len(result[1][0]) > 0 else np.zeros_like(input_beta).transpose((2, 0, 1))
                        result = np.sum(result[1][0], axis=0).astype(np.int64)
                    preds += result

                # majority voting based mask
                voted_pred = np.where(preds > len(configs) // 2, 255, 0)
                # averaged based mask
                # outputs.append(voted_pred)

                # averaged_pred = preds / len(models)
                # averaged_mask = np.where(averaged_pred >= 0.5, 255, 0)
                # apply morphological operation
                cv2.morphologyEx(np.uint8(voted_pred ), cv2.MORPH_OPEN, np.ones((open_disk_r, open_disk_r)))
                label_mask = label(voted_pred )
                regions = regionprops(label_mask)
                for props in regions:
                    area = props.area
                    e = props.eccentricity
                    if (e < 1) and (area >= 25):
                        # props.box returns in this order [min_row, min_col, max_row, max_col] => [y1, x1, y2, x2]
                        y1, x1, y2, x2 = list(map(lambda x: int(x), props.bbox))
                        centerCoord = bbox_to_cxy([x1, y1, x2, y2])
                        result = [image_name, centerCoord[0], centerCoord[1]]
                        result_analysis.append(result)

                # file_name = osp.join(img_path, image_name)
                # Image.fromarray(averaged_mask.astype(np.uint8)).save(file_name)

            if (i > 0 and i % 1000 == 0) or (i == len(fnames) - 1):
                print(TAG2, i, '[len(result_analysis)]', len(result_analysis))
                test_results_analysis_df = pd.DataFrame(data=result_analysis, columns=['image_id', 'x', 'y'])
                test_results_analysis_df.to_csv(df_path, index=False)

            prog_bar.update()

    # ONLY ENSEMBLED SEGMENTATION
    # build the dataloader
    else:
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=4,
            dist=False,
            shuffle=False,
        )
        
        models = load_models(configs, checkpoints)

        ensemble_outputs_npz = osp.join(tmpdir, 'model_ensemble2-outputs.npz')
        dataset = data_loader.dataset

        if not osp.exists(ensemble_outputs_npz):
            prog_bar = mmcv.ProgressBar(len(dataset))
            loader_indices = data_loader.batch_sampler

            for batch_indices, data in zip(loader_indices, data_loader):
                preds = 0
                for (model_type, model) in models:
                    x, _ = scatter_kwargs(inputs=data, kwargs=None, target_gpus=model.device_ids)
                    if args.aug_test:
                        logits = model.module.aug_test_logits(**x[0])
                    else:
                        # logits = model.module.simple_test_logits(**x[0])
                        result = model(return_loss=False, **data)
                    print(TAG2, 'shape:', result[0].shape, 'dtype:', result[0].dtype, 'unique', np.unique(result))
                    # pred = logits.argmax(axis=1).squeeze()
                    # pred = torch.sigmoid(logits[0,0])
                    # print(pred.shape())
                    preds += result
                # majority voting based mask
                # voted_pred = np.where(preds > len(configs) // 2, 255, 0)
                # averaged based mask
                averaged_pred = preds/len(models)
                averaged_mask = np.where(averaged_pred >= 0.5, 255, 0)
                # apply morphological operation
                cv2.morphologyEx(np.uint8(averaged_mask), cv2.MORPH_OPEN, np.ones((open_disk_r,open_disk_r)))
                #outputs.append(voted_pred)
                # combining all masks in a single array
                outputs.append(averaged_mask)
                # getting path to save image
                img_info = dataset.img_infos[batch_indices[0]]
                img_path = osp.join(tmpdir, 'imgs')
                if not osp.exists(img_path):
                    os.makedirs(img_path)
                file_name = osp.join(img_path, img_info['ann']['seg_map'].split(osp.sep)[-1])
                Image.fromarray(averaged_mask.astype(np.uint8)).save(file_name)
                prog_bar.update()
            outputs = np.array(outputs)
            print(TAG2, '[outputs]', type(outputs), len(outputs))
            print(TAG2, '[outputs]', outputs.shape, outputs.dtype)
            np.savez_compressed(ensemble_outputs_npz, outputs)
        else:
            outputs = np.load(ensemble_outputs_npz)['arr_0']
            # print('[outputs]', type(outputs), outputs.files)
            # outputs = outputs.files[0]
        print(TAG2, '[outputs]', type(outputs), len(outputs))
        print(TAG2, '[outputs]', outputs.shape, outputs.dtype)

        recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val = calculate('test', model_name, dataset, outputs)
        save(tmpdir, model_name, 'test', recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val)


def calculate(dataset_type, model_name, dataset, outputs):
    TAG = f'[{model_name}][calculate]'
    # thresholds = [0.25, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # print(TAG, f'[{dataset_type}][thresholds]', thresholds)
    # classes = self.cfg.classes
    # print(TAG, f'[{dataset_type}][classes]', classes)
    distance =  12
    recalls =       np.zeros(2)
    precisions =    np.zeros(2)
    fscores =       np.zeros(2)
    kappas =        np.zeros(2)
    accuracies =    np.zeros(2)
    # recall, precision, fscore, kappa, accuracy = -99, -99, -99, -99, -99
    counts_saved = []

    confusion_matrix_0, confusion_matrix_d = np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])
    kappa_gt_counts, kappa_dt_counts_0, kappa_dt_counts_d = [], [], []
    print(TAG, '[dataset.img_infos]', len(dataset.img_infos))
    for idx, image_info in enumerate(mmcv.track_iter_progress(dataset.img_infos)):
        gt_mask = dataset.get_gt_seg_map_by_idx(idx)
        dt_mask = outputs[idx]
        
        gt_contours, gt_count = lysto_utils.get_contours_from_mask(mask_image=gt_mask)
        dt_contours, dt_count1 = lysto_utils.get_contours_from_mask(mask_image=dt_mask)
        # if gt_count not in counts_saved:
        #     gt_output_path = osp.join(self.outputs_path[dataset_type], f'{idx}_gt.jpg')
        #     dt_output_path = osp.join(self.outputs_path[dataset_type], f'{idx}_dt_ep{self.epoch}.jpg')
        #     if not osp.exists(gt_output_path):
        #         plt.imsave(gt_output_path, gt_mask)
        #     plt.imsave(dt_output_path, dt_mask)
        #     counts_saved.append(gt_count)

        gt_bboxes = lysto_utils.get_bboxes_from_contours(gt_contours)
        dt_bboxes = lysto_utils.get_bboxes_from_contours(dt_contours)
        gt_count = len(gt_bboxes)
        dt_count1 = len(dt_bboxes)
        # if self.debug: print(TAG, '[gt_bboxes]', gt_bboxes.shape, gt_count)
        # if self.debug: print(gt_bboxes[:5])
        # if self.debug: print(TAG, '[dt_bboxes]', dt_bboxes.shape, dt_count1)
        # if self.debug: print(dt_bboxes[:5])
        kappa_gt_counts.append(gt_count)
        kappa_dt_counts_0.append(dt_count1)

        merged = []
        if distance > 0 and dt_count1 > 0:
            centroid_x = dt_bboxes[:, 0] + dt_bboxes[:, 2] / 2
            centroid_y = dt_bboxes[:, 1] + dt_bboxes[:, 3] / 2
            for j in range(0, dt_count1 - 1):
                if j not in merged:
                    for k in range(j + 1, dt_count1):
                        diff_x = np.square(np.float32(centroid_x[j] - centroid_x[k]))
                        diff_y = np.square(np.float32(centroid_y[j] - centroid_y[k]))
                        _distance = int(np.sqrt(diff_x + diff_y))
                        if _distance <= distance:
                            merged.append(k)
        dt_count2 = dt_count1 - len(merged)
        kappa_dt_counts_d.append(dt_count2)

        confusion_matrix_0[0][0] += min(gt_count, dt_count1)
        confusion_matrix_0[0][1] += np.abs(gt_count - dt_count1) if gt_count < dt_count1 else 0
        confusion_matrix_0[1][0] += np.abs(gt_count - dt_count1) if gt_count > dt_count1 else 0

        confusion_matrix_d[0][0] += min(gt_count, dt_count2)
        confusion_matrix_d[0][1] += np.abs(gt_count - dt_count2) if gt_count < dt_count2 else 0
        confusion_matrix_d[1][0] += np.abs(gt_count - dt_count2) if gt_count > dt_count2 else 0

    kappa_gt_counts = pd.Series(kappa_gt_counts)
    kappa_dt_counts_0 = pd.Series(kappa_dt_counts_0)
    kappa_dt_counts_d = pd.Series(kappa_dt_counts_d)
    for matrix_idx, (matrix, kappa_dt_counts) in enumerate([(confusion_matrix_0, kappa_dt_counts_0), (confusion_matrix_d, kappa_dt_counts_d)]):
        TP = matrix[0][0]
        FP = matrix[0][1]
        FN = matrix[1][0]

        recall = TP / (TP + FN)
        precision = TP / (TP + FP) if not np.isnan(TP / (TP + FP)) else 1
        fscore = 2 * precision * recall / (precision + recall)
        kappa = metrics.lysto_qwk(kappa_gt_counts, kappa_dt_counts)
        # kappa = -100
        accuracy = TP / (TP + FP + FN)

        recalls[matrix_idx] = recall
        precisions[matrix_idx] = precision
        fscores[matrix_idx] = fscore
        kappas[matrix_idx] = kappa
        accuracies[matrix_idx] = accuracy

    return recalls, precisions, fscores, kappas, accuracies


def save(path, model_name, dataset_type, recalls, precisions, fscores, kappas, accuracies):
    TAG = f'[{model_name}][save]'
    df_columns = ['distance', 'recall', 'precision', 'fscore', 'kappa', 'accuracy']
    df_metrics = pd.DataFrame(index=range(2), columns=df_columns, dtype=np.float32)
    df_metrics = df_metrics.fillna(0)

    idx = 0
    for distance_idx, distance in enumerate([0, 12]):
        df_metrics.iloc[idx, 0] = distance
        df_metrics.iloc[idx, 1] = np.average(recalls[distance_idx])
        df_metrics.iloc[idx, 2] = np.average(precisions[distance_idx])
        df_metrics.iloc[idx, 3] = np.average(fscores[distance_idx])
        df_metrics.iloc[idx, 4] = np.average(kappas[distance_idx])
        df_metrics.iloc[idx, 5] = np.average(accuracies[distance_idx])
        idx += 1
    
    df_path = osp.join(path, f'{model_name}-{dataset_type}.csv')
    # df_metrics = pd.read_csv(df_path) if osp.exists(df_path) else None
    # df_metrics = pd.concat([df_metrics, df_metrics], axis=0, ignore_index=True)
    df_metrics.to_csv(df_path, index=False)
    print('----------------------------------')
    print(TAG, 'df_metrics saved at:', df_path)
    print(df_metrics)
    print('----------------------------------')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model Ensemble with logits result')
    parser.add_argument(
        '--config', type=str, nargs='+', help='ensemble config files path')
    parser.add_argument(
        '--model_name', type=str, default='ensemble', help='ensemble model name')
    parser.add_argument(
        '--checkpoint',
        type=str,
        nargs='+',
        help='ensemble checkpoint files path')
    parser.add_argument(
        '--aug-test',
        action='store_true',
        help='control ensemble aug-result or single-result (default)')
    parser.add_argument(
        '--out', type=str, default='ensemble_results', help='the dir to save result')
    parser.add_argument(
        '--gpus', type=int, nargs='+', default=[0], help='id of gpu to use')
    parser.add_argument(
        '--testpath', type=str, help='test images path in case of external test')
    parser.add_argument(
        '--class_confg', type=str, default=None)
    parser.add_argument(
        '--class_checkpoint', type=str, default=None)

    args = parser.parse_args()
    assert len(args.config) == len(args.checkpoint), \
        f'len(config) must equal len(checkpoint), ' \
        f'but len(config) = {len(args.config)} and' \
        f'len(checkpoint) = {len(args.checkpoint)}'
    assert args.out, "ensemble result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    print()
