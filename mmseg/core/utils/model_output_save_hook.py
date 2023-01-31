import os
import numpy as np

import torch

import mmcv
from mmcv import Config
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.core import encode_mask_results

@HOOKS.register_module(name='ModelOutputSaveHook', force=True)
class ModelOutputSaveHook(Hook):
    def __init__(self, save_interval, file_prefix, path_config, base_dir):
        self.hook_name = 'ModelOutputSaveHook'
        self.save_interval = save_interval
        self.file_prefix = file_prefix
        self.cfg = Config.fromfile(path_config)
        self.cfg.data.test.ann_file = os.path.join(self.cfg.PATH_DATASET, 'test_mask_images1-sample.json')
        self.cfg.data.samples_per_gpu = 5
        self.cfg.data.workers_per_gpu = 5
        self.base_dir = os.path.join(self.cfg.work_dir, base_dir)
        mmcv.mkdir_or_exist(self.base_dir)
        # self.dataloader_val = self._build_dataloader(self.cfg.data, self.cfg.data.val)
        self.dataloader_test = self._build_dataloader(self.cfg.data, self.cfg.data.test)

    def _build_dataloader(self, cfg_data, data_type):
        if isinstance(data_type, dict):
            data_type.test_mode = True
        elif isinstance(data_type, list):
            for ds_cfg in data_type:
                ds_cfg.test_mode = True
        if cfg_data.samples_per_gpu > 1:
            data_type.pipeline = replace_ImageToTensor(data_type.pipeline)
        dataset = build_dataset(data_type)
        print('[_build_dataloader][dataset]', len(dataset))
        # print(dir(dataset))
        data_loader = build_dataloader(dataset, samples_per_gpu=cfg_data.samples_per_gpu, workers_per_gpu=cfg_data.workers_per_gpu, dist=False, shuffle=False)
        return data_loader

    def single_gpu_test(self, model, data_loader):
        model.eval()
        results = []
        filenames = []
        dataset = data_loader.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)

            batch_size = len(result)
            filename = [single_img_data['filename'] for single_img_data in data['img_metas'][0].data[0]]

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            results.extend(result)
            filenames.extend(filename)

            for _ in range(batch_size):
                prog_bar.update()
        return results, filenames

    def run_for_sepcific_epoch(self, model, epoch=0):
        self.epoch = epoch
        print(f'[{self.hook_name}] epoch={self.epoch}')
        self._save_model_output(model)

    # def before_epoch(self, runner):
    #     self.epoch = runner.epoch + 1
    #     print(f'[{self.hook_name}] epoch={self.epoch}')
    #     if self.epoch == 1:
    #         self._save_model_output(runner.model)

    def after_epoch(self, runner):
        self.epoch = runner.epoch + 1
        print(f'[{self.hook_name}] epoch={self.epoch}')
        if self.epoch % self.save_interval == 0 or self.epoch == self.cfg.total_epochs:
            self._save_model_output(runner.model)

    def _save_model_output(self, model):
        TAG = f'[{self.hook_name}][_save_model_output]'
        # # validation dataset
        # outputs_val = single_gpu_test(model, self.dataloader_val, show=False)
        # print('\n', TAG, f'[outputs_val]', len(outputs_val))
        # thresholds_val, recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val = self._calculate('val', self.dataloader_val.dataset, outputs_val)
        # self._save('val', thresholds_val, recalls_val, precisions_val, fscores_val, kappas_val, accuracies_val)

        # test dataset
        outputs_test, filenames_test = self.single_gpu_test(model, self.dataloader_test)
        self._save_to_disk(outputs_test, filenames_test)

    def _save_to_disk(self, outputs, filenames, datatype):
        path_filenames = os.path.join(self.base_dir, f'{self.file_prefix}-filenames_{datatype}.csv')
        print('[path_filenames]', path_filenames)
        with open(path_filenames, 'w') as file:
            file.write('filenames_test\n')
            for f in filenames:
                file.write(f + '\n')

        path_output = os.path.join(self.base_dir, f'{self.file_prefix}-outputs_{datatype}.npz')
        print('[path_output]', path_output)
        outputs = []
        for output in outputs:
            # convert float32 to float16
            output[0][0] = output[0][0].astype(np.float16)
            # convert python list to numpy array
            output[1][0] = np.array(output[1][0])
            # append this output
            outputs.append(output)
        
        # convert python list to numpy array
        outputs = np.asarray(outputs)
        # save the model's predictions as numpy compressed array
        np.savez_compressed(path_output, outputs)
