import os
import time
import numpy as np
import pandas as pd
from . import ml_metrics as metrics
import matplotlib.pyplot as plt

import mmcv
from mmcv import Config
from mmcv.runner import HOOKS, Hook
from mmseg.apis import single_gpu_test, multi_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import lysto_utils

@HOOKS.register_module(name='LymphCountEvalHook', force=True)
class LymphCountEvalHook(Hook):
    def __init__(self, eval_interval, file_prefix, path_config, base_dir, fold=-1, debug=False):
        self.hook_name = 'LymphCountEvalHook'
        self.eval_interval = eval_interval
        self.file_prefix = file_prefix
        self.fold = fold
        self.debug = debug
        self.cfg = Config.fromfile(path_config)
        if self.fold > 0:
            self.cfg.work_dir = f'trained_models/{self.cfg.DATASET}-models/{self.cfg.MODEL_NAME}/setting{self.cfg.S}/fold{self.fold}/'
            self.cfg.data.train.ann_file = f'{self.cfg.PATH_DATASET}/cross-validation-f5-test-coco/fold_{self.fold}_train.json'
            self.cfg.data.val.ann_file = f'{self.cfg.PATH_DATASET}/cross-validation-f5-test-coco/fold_{self.fold}_val.json'
            self.cfg.data.test.ann_file = f'{self.cfg.PATH_DATASET}/cross-validation-f5-test-coco/fold_{self.fold}_val.json'
        self.cfg.data.samples_per_gpu = 16
        self.cfg.data.workers_per_gpu = 8
        self.base_dir = os.path.join(self.cfg.work_dir, base_dir)
        self.outputs_path = {
            'val': os.path.join(self.base_dir, 'outputs_val'), 
            'test': os.path.join(self.base_dir, 'outputs_test')
        }
        mmcv.mkdir_or_exist(self.base_dir)
        mmcv.mkdir_or_exist(self.outputs_path['val'])
        mmcv.mkdir_or_exist(self.outputs_path['test'])
        self.dataloader_val = self._build_dataloader(self.cfg.data, self.cfg.data.val)
        self.dataloader_test = self._build_dataloader(self.cfg.data, self.cfg.data.test)

    def _build_dataloader(self, cfg_data, data_type):
        print(f'[{self.hook_name}][_build_dataloader] data_type.ann_dir={data_type.ann_dir}')
        if isinstance(data_type, dict):
            data_type.test_mode = True
        elif isinstance(data_type, list):
            for ds_cfg in data_type:
                ds_cfg.test_mode = True
        # if cfg_data.samples_per_gpu > 1:
        #     data_type.pipeline = replace_ImageToTensor(data_type.pipeline)
        dataset = build_dataset(data_type)
        data_loader = build_dataloader(dataset, samples_per_gpu=cfg_data.samples_per_gpu, workers_per_gpu=cfg_data.workers_per_gpu, dist=False, shuffle=False)
        return data_loader

    def eval_specific_epoch(self, model, epoch=0):
        self.epoch = epoch
        print(f'[{self.hook_name}][eval_specific_epoch] epoch={self.epoch}')
        self._perform_evaluation(model)

    # def before_epoch(self, runner):
    #     self.epoch = runner.epoch + 1
    #     # print(f'[{self.hook_name}] epoch={self.epoch}')
    #     # if self.epoch == 1:
    #     #     self._perform_evaluation(runner.model)

    def after_train_epoch(self, runner):
        self.epoch = runner.epoch + 1
        print(f'[{self.hook_name}][after_train_epoch] epoch={self.epoch}, runner.epoch={runner.epoch}')
        if self.epoch % self.eval_interval == 0 or self.epoch == self.cfg.total_epochs:
            self._perform_evaluation(runner.model)

    def _perform_evaluation(self, model):
        TAG = f'[{self.hook_name}][_perform_evaluation]'

        # try:
        # val dataset
        outputs_val = single_gpu_test(model, self.dataloader_val)
        print('\n', TAG, f'[outputs_val]', len(outputs_val))
        recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val = self._calculate('val', self.dataloader_val.dataset, outputs_val)
        self._save('val', recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val)
        # except Exception as e:
        #     print('\n', TAG, '[error evaluating val dataset]')

        # try:
        # test dataset
        outputs_test = single_gpu_test(model, self.dataloader_test)
        print('\n', TAG, f'[outputs_test]', len(outputs_test))
        recalls_test, precisions_test, fscores_test, kappas_test, accuracies_test = self._calculate('test', self.dataloader_test.dataset, outputs_test)
        self._save('test', recalls_test, precisions_test, fscores_test, kappas_test, accuracies_test)
        # except Exception as e:
        #     print('\n', TAG, '[error evaluating val dataset]')

    def _calculate(self, dataset_type, dataset, outputs):
        TAG = f'[{self.hook_name}][_calculate]'
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
            if gt_count not in counts_saved:
                gt_output_path = os.path.join(self.outputs_path[dataset_type], f'{idx}_gt.jpg')
                dt_output_path = os.path.join(self.outputs_path[dataset_type], f'{idx}_dt_ep{self.epoch}.jpg')
                if not os.path.exists(gt_output_path):
                    plt.imsave(gt_output_path, gt_mask)
                plt.imsave(dt_output_path, dt_mask)
                counts_saved.append(gt_count)

            gt_bboxes = lysto_utils.get_bboxes_from_contours(gt_contours)
            dt_bboxes = lysto_utils.get_bboxes_from_contours(dt_contours)
            gt_count = len(gt_bboxes)
            dt_count1 = len(dt_bboxes)
            if self.debug: print(TAG, '[gt_bboxes]', gt_bboxes.shape, gt_count)
            if self.debug: print(gt_bboxes[:5])
            if self.debug: print(TAG, '[dt_bboxes]', dt_bboxes.shape, dt_count1)
            if self.debug: print(dt_bboxes[:5])
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

    def _save(self, dataset_type, recalls, precisions, fscores, kappas, accuracies):
        TAG = f'[{self.hook_name}][_save]'
        df_columns = ['epoch', 'distance', 'recall', 'precision', 'fscore', 'kappa', 'accuracy']
        df_current_epoch = pd.DataFrame(index=range(2), columns=df_columns, dtype=np.float32)
        df_current_epoch = df_current_epoch.fillna(0)
        df_current_epoch.epoch = self.epoch

        idx = 0
        for distance_idx, distance in enumerate([0, 12]):
            df_current_epoch.iloc[idx, 1] = distance
            df_current_epoch.iloc[idx, 2] = np.average(recalls[distance_idx])
            df_current_epoch.iloc[idx, 3] = np.average(precisions[distance_idx])
            df_current_epoch.iloc[idx, 4] = np.average(fscores[distance_idx])
            df_current_epoch.iloc[idx, 5] = np.average(kappas[distance_idx])
            df_current_epoch.iloc[idx, 6] = np.average(accuracies[distance_idx])
            idx += 1
        
        df_path = os.path.join(self.base_dir, f'{self.file_prefix}-{dataset_type}.csv')
        df_metrics = pd.read_csv(df_path) if os.path.exists(df_path) else None
        df_metrics = pd.concat([df_metrics, df_current_epoch], axis=0, ignore_index=True)
        df_metrics.to_csv(df_path, index=False)
        print('----------------------------------')
        print(TAG, 'df_metrics saved at:', df_path)
        print(df_current_epoch)
        print('----------------------------------')
