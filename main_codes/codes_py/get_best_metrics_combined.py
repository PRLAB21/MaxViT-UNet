import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG = '[z-get_best_metrics]'
dataset_name = 'LYSTO'
threshold_types = ['threshold=0.25', 'threshold=0.50', 'threshold=0.75', 'threshold=0.95', 'threshold=0.5:0.95']
for i in [11, 12, 13, 14, 15]:
    base_path = f'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting{i}'
    for t in ['test', 'val']:
        csv_path = os.path.join(base_path, 'combined', 'eval1_mask3-test-combined', f'maskrcnn-lymphocytenet3-cm1-{t}.csv')
        print(csv_path, os.path.exists(csv_path))
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='epoch')
        # df = df.loc[df.epoch.isin(list(range(2, 21, 2)))]

        df_best_metrics = pd.DataFrame()
        for d in [0, 12]:
            for threshold_idx, threshold_type in enumerate(threshold_types):
                condition = (df.distance == d) & (df.threshold == threshold_type)
                # print(f'[{key}]\n', df.loc[condition])
                idx_max_fscore = df.loc[condition, 'fscore'].idxmax()
                # print('[idx_max_fscore]', idx_max_fscore)
                selected_row = df.loc[idx_max_fscore]
                # selected_row['model_config'] = key
                # print('[selected_row]\n', selected_row)
                df_best_metrics = df_best_metrics.append(selected_row)
        # df_best_metrics = df_best_metrics[column_order]
        print(TAG, '[df_best_metrics]\n', df_best_metrics)
        # df_best_metrics.to_csv('z-get_best_metrics_fold_wise.csv', index=False)
    print()
