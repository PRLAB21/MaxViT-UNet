import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

MMSEG_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/mmclassification')
DATASET_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/datasets')

json_path = MMSEG_HOME_PATH / 'trained_models/lymph_vs_nolymph/lysto_models/lympnet2/setting2/20221101_235923.log.json'
csv_path = MMSEG_HOME_PATH / 'trained_models/lymph_vs_nolymph/lysto_models/lympnet2/setting2/lympnet2-s2-metrics.csv'
csv_columns = ['epoch', 'accuracy', 'recall', 'precision', 'f1_score', 'loss']
json_file_data = open(json_path, 'r').readlines()
df_metrics = pd.DataFrame(columns=csv_columns)
metrics_row = pd.Series(index=csv_columns)

for i, line in enumerate(json_file_data):
    json_data = json.loads(line)
    if 'mode' in json_data and json_data['mode'] == 'val':
        if json_data['epoch'] % 2 == 0:
            metrics_row['epoch'] = json_data['epoch']
            
            if 'loss' in json_data:
                metrics_row['loss'] = json_data['loss']
                df_metrics = pd.concat([df_metrics, metrics_row.to_frame().T], ignore_index=True)
            else:
                metrics_row['accuracy'] = json_data['accuracy']
                metrics_row['recall'] = json_data['recall']
                metrics_row['precision'] = json_data['precision']
                metrics_row['f1_score'] = json_data['f1_score']

df_metrics['epoch'] = df_metrics['epoch'].astype(np.int32)
print('[df_metrics]\n', df_metrics)
df_metrics.to_csv(csv_path, index=False)
