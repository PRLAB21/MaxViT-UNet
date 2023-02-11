# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from mmseg.core.utils import ml_metrics as metrics
import matplotlib.pyplot as plt
import mmcv
import cv2
import numpy as np
import pandas as pd
import torch
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter_kwargs
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import lysto_utils

@torch.no_grad()
def main(args):

    models = []
    gpu_ids = args.gpus
    configs = args.config
    ckpts = args.checkpoint
  
    model_name = args.model_name

    cfg = mmcv.Config.fromfile(configs[0])

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
        ]
        cfg.data.test.pipeline[1].flip = True
    else:
        cfg.data.test.pipeline[1].img_ratios = [1.0]
        cfg.data.test.pipeline[1].flip = False

    torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=4,
        dist=False,
        shuffle=False,
    )

    for idx, (config, ckpt) in enumerate(zip(configs, ckpts)):
        cfg = mmcv.Config.fromfile(config)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        if cfg.get('fp16', None):
            wrap_fp16_model(model)
        load_checkpoint(model, ckpt, map_location='cpu')
        torch.cuda.empty_cache()
        tmpdir = args.out
        mmcv.mkdir_or_exist(tmpdir)
        model = MMDataParallel(model, device_ids=[gpu_ids[idx % len(gpu_ids)]])
        model.eval()
        models.append(model)

    ensemble_outputs_npz = os.path.join(tmpdir, 'model_ensemble2-outputs2.npz')
    dataset = data_loader.dataset

    if not os.path.exists(ensemble_outputs_npz):
        prog_bar = mmcv.ProgressBar(len(dataset))
        loader_indices = data_loader.batch_sampler
        outputs = []
        open_disk_r = 10
        
        for batch_indices, data in zip(loader_indices, data_loader):
            preds = 0
            for model in models:
                x, _ = scatter_kwargs(inputs=data, kwargs=None, target_gpus=model.device_ids)
                if args.aug_test:
                    logits = model.module.aug_test_logits(**x[0])
                else:
                    # logits = model.module.simple_test_logits(**x[0])
                    result = model(return_loss=False, **data)
                print('shape:', result[0].shape, 'dtype:', result[0].dtype, 'unique', np.unique(result))
                #pred = logits.argmax(axis=1).squeeze()
                #pred = torch.sigmoid(logits[0,0])
                #print(pred.shape())
                preds += result
            # majority voting based mask
            # voted_pred = np.where(preds > len(configs) // 2, 255, 0)
            # averaged based mask
            averged_pred = preds/len(models)
            averaged_mask = np.where(averged_pred >= 0.5, 255, 0)
            # apply morphological operation
            cv2.morphologyEx(np.uint8(averaged_mask), cv2.MORPH_OPEN, np.ones((open_disk_r,open_disk_r)))
            #outputs.append(voted_pred)
            # combining all masks in a single array
            outputs.append(averaged_mask)
            # getting path to save image
            img_info = dataset.img_infos[batch_indices[0]]
            img_path = os.path.join(tmpdir, 'imgs')
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            file_name = os.path.join(img_path, img_info['ann']['seg_map'].split(os.path.sep)[-1])
            Image.fromarray(averaged_mask.astype(np.uint8)).save(file_name)
            prog_bar.update()
        outputs = np.array(outputs)
        print('[outputs]', type(outputs), len(outputs))
        print('[outputs]', outputs.shape, outputs.dtype)
        np.savez_compressed(ensemble_outputs_npz, outputs)
    else:
        outputs = np.load(ensemble_outputs_npz)['arr_0']
        # print('[outputs]', type(outputs), outputs.files)
        # outputs = outputs.files[0]
    print('[outputs]', type(outputs), len(outputs))
    print('[outputs]', outputs.shape, outputs.dtype)

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
        #     gt_output_path = os.path.join(self.outputs_path[dataset_type], f'{idx}_gt.jpg')
        #     dt_output_path = os.path.join(self.outputs_path[dataset_type], f'{idx}_dt_ep{self.epoch}.jpg')
        #     if not os.path.exists(gt_output_path):
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
    
    df_path = os.path.join(path, f'{model_name}-{dataset_type}.csv')
    # df_metrics = pd.read_csv(df_path) if os.path.exists(df_path) else None
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

    args = parser.parse_args()
    assert len(args.config) == len(args.checkpoint), \
        f'len(config) must equal len(checkpoint), ' \
        f'but len(config) = {len(args.config)} and' \
        f'len(checkpoint) = {len(args.checkpoint)}'
    assert args.out, "ensemble result out-dir can't be None"
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print()
