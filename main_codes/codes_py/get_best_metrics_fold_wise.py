import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAG = '[z-get_best_metrics]'
dataset_name = 'LYSTO'
threshold_types = ['threshold=0.25', 'threshold=0.50', 'threshold=0.75', 'threshold=0.95', 'threshold=0.5:0.95']
df_combined = []
base_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting15'
for f in range(5):
    csv_path = os.path.join(base_path, f'fold{f+1}', 'eval1_mask3', 'maskrcnn-lymphocytenet3-cm1-val.csv')
    print(csv_path, os.path.exists(csv_path))
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='epoch')
    df = df.loc[df.epoch.isin(list(range(2, 21, 2)))]
    df['fold'] = f + 1
    # print('[df]\n', df)
    df_combined.append(df)
df_combined = pd.concat(df_combined).reset_index(drop=True)
print('[df_combined]\n', df_combined)

df_best_metrics = pd.DataFrame()
for d in [0, 12]:
    for threshold_idx, threshold_type in enumerate(threshold_types):
        condition = (df_combined.distance == d) & (df_combined.threshold == threshold_type)
        # print(f'[{key}]\n', df_combined.loc[condition])
        idx_max_fscore = df_combined.loc[condition, 'fscore'].idxmax()
        # print('[idx_max_fscore]', idx_max_fscore)
        selected_row = df_combined.loc[idx_max_fscore]
        # selected_row['model_config'] = key
        # print('[selected_row]\n', selected_row)
        df_best_metrics = df_best_metrics.append(selected_row)
# df_best_metrics = df_best_metrics[column_order]
print(TAG, '[df_best_metrics]\n', df_best_metrics)
# df_best_metrics.to_csv('z-get_best_metrics_fold_wise.csv', index=False)
