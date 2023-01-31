import os
import numpy as np
import pandas as pd

import torch

from mmcv import Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'running on {device}')

def merge_eval_csvs():
    for path_config in path_configs:
        dfs = []
        print('[path_config]', path_config)
        cfg = Config.fromfile(path_config)
        path_merge_csv = os.path.join(f'./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/eval1_mask3-{cfg.MODEL_NAME}-s{cfg.S}-val.csv')
        if os.path.exists(path_merge_csv):
            print(TAG, 'path_merge_csv already exists:', path_merge_csv)
            continue

        for fold in range(1, 6):
            # print(TAG, '[fold]', fold)
            cfg.work_dir = f'./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/fold{fold}/'
            # print(TAG, '[cfg.work_dir]', cfg.work_dir)
            path_eval_csv = os.path.join(cfg.work_dir, 'eval1_mask3', f'{cfg.MODEL_NAME}-val.csv')
            print(TAG, '[path_eval_csv]', os.path.exists(path_eval_csv), path_eval_csv)
            df_eval_csv = pd.read_csv(path_eval_csv)
            df_eval_csv = df_eval_csv.loc[df_eval_csv.epoch % 2 == 0]
            dfs.append(df_eval_csv)
            print(TAG, '[df_eval_csv]', df_eval_csv.shape)
            # print('-' * 50)

        df_merged = dfs[0].copy()
        df_merged = df_merged.fillna(-1)
        columns_to_avg = ['recall', 'precision', 'fscore', 'kappa', 'accuracy']
        # print('[df_merged]\n', df_merged)
        print('[columns]', dfs[0].columns)
        for col in columns_to_avg:
            # print(col)
            df_merged[col] = np.mean([df[col].values for df in dfs], axis=0)
        print('[df_merged]\n', df_merged)
        df_merged.to_csv(path_merge_csv, index=False)

def merge_stats_csvs():
    for path_config in path_configs:
        dfs = []
        print(TAG, '[path_config]', path_config)
        path_merge_csv = f'./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/statistics_merged-{cfg.MODEL_NAME}-s{cfg.S}.csv'
        if os.path.exists(path_merge_csv):
            print(TAG, 'path_merge_csv exists:', path_merge_csv)
            continue

        for fold in range(1, 11):
            print(TAG, '[fold]', fold)
            cfg = Config.fromfile(path_config)
            cfg.work_dir = f'./trained_models/{cfg.DATASET}-models/{cfg.MODEL_NAME}/setting{cfg.S}/fold{fold}/'
            print(TAG, '[cfg.work_dir]', cfg.work_dir)
            path_statistics_csv = os.path.join(cfg.work_dir, 'final_model_script', f'statistics-{cfg.MODEL_NAME}-s{cfg.S}-f{fold}.csv')
            print(TAG, '[path_statistics_csv]', os.path.exists(path_statistics_csv), path_statistics_csv)
            df_statistics_csv = pd.read_csv(path_statistics_csv)
            dfs.append(df_statistics_csv)
            print(TAG, '[df_statistics_csv]\n', df_statistics_csv)
            print('-' * 50)

        dfs = pd.concat(dfs)
        dfs.to_csv(path_merge_csv, index=False)

# define global variables
TAG = '[z-merge_cross_val_folds]'
path_configs = [
    './configs/lysto/maskrcnn_lymphocytenet3_cm1_s11_lysto.py',
    './configs/lysto/maskrcnn_lymphocytenet3_cm1_s12_lysto.py',
    './configs/lysto/maskrcnn_lymphocytenet3_cm1_s13_lysto.py',
    './configs/lysto/maskrcnn_lymphocytenet3_cm1_s14_lysto.py',
    './configs/lysto/maskrcnn_lymphocytenet3_cm1_s15_lysto.py',
]
