import os
import numpy as np
import pandas as pd
from . import panoptic_quality as pq_metric

import mmcv
from mmcv import Config
from mmcv.runner import HOOKS, Hook
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset

@HOOKS.register_module(name='MoNuSAC_EvalHook', force=True)
class MoNuSAC_EvalHook(Hook):
    def __init__(self, eval_interval, file_prefix, path_config, base_dir, debug=False):
        self.hook_name = 'MoNuSAC_EvalHook'
        self.eval_interval = eval_interval
        self.file_prefix = file_prefix
        self.debug = debug
        self.cfg = Config.fromfile(path_config)
        self.cfg.data.samples_per_gpu = 4
        self.cfg.data.workers_per_gpu = 4
        self.base_dir = os.path.join(self.cfg.work_dir, base_dir)
        mmcv.mkdir_or_exist(self.base_dir)
        self.dataloader_val = self._build_dataloader(self.cfg.data, self.cfg.data.val)
        self.best_score = 0
        self.best_ckpt_path = None

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

    # def after_train_iter(self, runner):
    #     self.epoch = runner.epoch + 1
    #     print(f'[{self.hook_name}][after_train_epoch] epoch={self.epoch}, runner.epoch={runner.epoch}')
    #     if runner.iter > 100:
    #         self._perform_evaluation(runner.model)

    def after_train_epoch(self, runner):
        self.epoch = runner.epoch + 1
        print(f'[{self.hook_name}][after_train_epoch] epoch={self.epoch}, runner.epoch={runner.epoch}')
        if self.epoch % self.eval_interval == 0 or self.epoch == self.cfg.total_epochs:
            current_PQ = self._perform_evaluation(runner.model)
            self._save_ckpt(runner, current_PQ)

    def _perform_evaluation(self, model):
        TAG = f'[{self.hook_name}][_perform_evaluation]'

        outputs_val = single_gpu_test(model, self.dataloader_val)
        print('\n', TAG, f'[outputs_val]', len(outputs_val))
        dataset = self.dataloader_val.dataset
        PQs = []

        for idx, _ in enumerate(mmcv.track_iter_progress(dataset.img_infos)):
            gt_mask = dataset.get_gt_seg_map_by_idx(idx)
            dt_mask = outputs_val[idx]
            image_PQ = pq_metric.panoptic_quality(gt_mask, dt_mask)
            PQs.append(image_PQ)
        current_PQ = np.mean(PQs)

        df_path = os.path.join(self.base_dir, f'{self.file_prefix}-val.csv')
        df_metrics = pd.read_csv(df_path) if os.path.exists(df_path) else None
        df_current_epoch = pd.DataFrame({'epoch': [self.epoch], 'PQ': [current_PQ]})
        df_metrics = pd.concat([df_metrics, df_current_epoch], axis=0, ignore_index=True)
        df_metrics.to_csv(df_path, index=False)

        print('----------------------------------')
        print(TAG, 'df_metrics saved at:', df_path)
        print(df_current_epoch)
        print('----------------------------------')

        return current_PQ

    def _save_ckpt(self, runner, current_PQ):
        if current_PQ > self.best_score:
            self.best_score = current_PQ

            if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                os.remove(self.best_ckpt_path)
                runner.logger.info(f'The previous best checkpoint {self.best_ckpt_path} was removed')

            best_ckpt_name = f'best_PQ_e{self.epoch:03}.pth'
            self.best_ckpt_path = os.path.join(self.base_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(self.base_dir, filename_tmpl=best_ckpt_name, create_symlink=False)
            runner.logger.info(f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(f'Best PQ is {self.best_score:0.4f} at {self.epoch} epoch.')
