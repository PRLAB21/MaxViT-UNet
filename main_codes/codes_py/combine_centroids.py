from operator import index
import os
import glob
import numpy as np
import pandas as pd
from pprint import pprint

base_path = 'trained_models/lyon-models/maskrcnn-lymphocytenet3-cm1/setting6'
centroid_files = [f.split('/')[-1] for f in glob.glob(os.path.join(base_path, 'lyon-split*.csv'))]
centroid_files = sorted(centroid_files, key=lambda x: int(x.split('-')[1][5:]))
pprint(centroid_files)

df_centroids = []
for filepath in centroid_files:
    df = pd.read_csv(os.path.join(base_path, filepath))
    df['dataset_split'] = int(filepath.split('-')[1][5:])
    rows = df['confidence_score'] >= 0.5
    df = df.loc[rows]
    # print(df['confidence_score'].min(), df['confidence_score'].max())
    # df = df.drop(labels=['id', 'confidence_score'], axis=1)
    print(df.shape)
    df_centroids.append(df)

df_combined = pd.concat(df_centroids)
print(df_combined.shape)
df_combined.to_csv(os.path.join(base_path, 'lyon-centroids-maskrcnn-lymphocytenet3-cm1-s6-ep30.csv'), index=None)
