import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import glob

MMSEG_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation')
DATASET_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/datasets')

def log_json2csv_metrics(json_path, csv_path):
    csv_columns = ['epoch', 'accuracy', 'recall', 'precision', 'f1_score', 'loss']
    json_file_data = open(json_path, 'r').readlines()
    df_metrics = pd.DataFrame(columns=csv_columns)
    metrics_row = pd.Series(index=csv_columns, dtype=np.float64)

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

path_lysto_models = MMSEG_HOME_PATH/'trained_models/lysto/'
for log_json_path in glob.glob('**/*.log.json', root_dir=str(path_lysto_models), recursive=True):
    print('[log_json_path]', log_json_path)
    log_json_filename = os.path.basename(log_json_path)
    # print('[log_json_filename]', log_json_filename)
    log_csv_filename = log_json_filename[:-4] + 'csv'
    # print('[log_csv_filename]', log_csv_filename)
    # print(os.path.dirname(log_json_path))
    log_csv_path = path_lysto_models / os.path.dirname(log_json_path) / log_csv_filename
    print('[log_csv_path] ', log_csv_path)
    log_json_path = path_lysto_models / log_json_path
    print('[log_json_path]', log_json_path)
    log_json2csv_metrics(log_json_path, log_csv_path)

# json_path = MMSEG_HOME_PATH / 'trained_models/lymph_vs_nolymph/lysto_models/lympnet2/setting2-2022-12-05-13-47/20221101_235923.log.json'
# csv_path = MMSEG_HOME_PATH / 'trained_models/lymph_vs_nolymph/lysto_models/lympnet2/setting2/lympnet2-s2-metrics.csv'
