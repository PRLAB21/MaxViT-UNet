import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import glob

MMSEG_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation')
DATASET_HOME_PATH = Path('/home/gpu02/maskrcnn-lymphocyte-detection/datasets')

path_lysto_models = MMSEG_HOME_PATH/'trained_models/lysto/'
path_tf_runs = MMSEG_HOME_PATH/'tf_runs/lysto'
os.makedirs(path_tf_runs, exist_ok=True)
for tf_logs_path in glob.glob('**/tf_logs', root_dir=str(path_lysto_models), recursive=True):
    log_file_names = os.listdir(path_lysto_models/tf_logs_path)
    # print('[tf_logs_path]', tf_logs_path, len(log_file_names))
    tf_logs_out_path = path_tf_runs/os.path.split(tf_logs_path)[0]
    os.makedirs(tf_logs_out_path, exist_ok=True)
    # print('[tf_logs_out_path]', tf_logs_out_path)
    for log_filename in log_file_names:
        src_filepath = path_lysto_models/tf_logs_path/log_filename
        dst_filepath = tf_logs_out_path/log_filename
        print('[src_filepath]', src_filepath)
        print('[dst_filepath]', dst_filepath)
        shutil.copy(src_filepath, dst_filepath)
