import os
import numpy as np
import pandas as pd

def extract_counts_from_statistics_csv():
    base_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting14/combined/inference_lysto_12000/'
    df_statistics = pd.read_csv(os.path.join(base_path, 'statistics-maskrcnn-lymphocytenet3-cm1-s14.csv'))
    print('[df_statistics]\n', df_statistics.shape)
    # df_statistics = df_statistics.drop(['threshold', 'image_name', 'gt_count'], axis=1)
    df_statistics = df_statistics.loc[df_statistics.distance == 12, ['threshold', 'image_name', 'dt_count']]
    print('[df_statistics]\n', df_statistics.shape)
    df_statistics['id'] = df_statistics['image_name'].apply(lambda x: int(x[5:-4]))
    df_statistics = df_statistics.rename(columns={'dt_count': 'count'})
    print('[df_statistics]\n', df_statistics.shape)

    for threshold in df_statistics.threshold.unique():
        df_output = df_statistics.loc[df_statistics.threshold == threshold, ['id', 'count']]
        print(threshold, df_output.shape)
        df_output.to_csv(os.path.join(base_path, f'predictions_t{int(threshold*100)}.csv'), index=False)

def extract_counts_from_inference_csv():
    base_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting14/combined/inference_pipeline_lysto/'
    df_inference = pd.read_csv(os.path.join(base_path, 'LymphNet-maskrcnn-lymphocytenet3-cm1-s14-2.csv'))
    print('[df_inference]\n', df_inference)
    # df_inference = df_inference.drop(['threshold', 'image_name', 'gt_count'], axis=1)
    df_counts = df_inference['image_id'].apply(lambda x: int(x[:-4])) \
                                        .value_counts() \
                                        .to_frame('count') \
                                        .reset_index() \
                                        .rename(columns={'index': 'id'}) \
                                        .sort_values(by='id')
    print('[df_counts]\n', df_counts)
    df_counts.to_csv(os.path.join(base_path, f'predictions.csv'), index=False)
    # df_inference['id'] = df_inference['image_name'].apply(lambda x: int(x[5:-4]))
    # df_inference = df_inference.rename(columns={'dt_count': 'count'})
    # print('[df_inference]\n', df_inference.shape)

    # for threshold in df_inference.threshold.unique():
    #     df_output = df_inference.loc[df_inference.threshold == threshold, ['id', 'count']]
    #     print(threshold, df_output.shape)
    #     df_output.to_csv(os.path.join(base_path, f'predictions_t{int(threshold*100)}.csv'), index=False)

extract_counts_from_statistics_csv()
# extract_counts_from_inference_csv()























# df_artifacts = df_statistics.loc[df_statistics.class_label == 'Artifact', 'image_id'].value_counts()
# print('[df_artifacts]\n', df_artifacts)
# print('[df_artifacts]', len(df_artifacts))

# df_counts[df_counts.index.isin(df_artifacts.index)] = 0
# df_counts = df_counts.to_frame().reset_index()
# df_counts = df_counts.rename(columns={'index': 'id', 'image_id': 'count'})
# df_counts['id'] = df_counts['id'].apply(lambda x: int(x[5:-4]))
# df_counts = df_counts.sort_values(by='id')
# print('[zzz]\n', df_counts)
# df_counts.to_csv(os.path.join(base_path, 'predictions.csv'), index=False)
# # df_results2 = df_statistics.groupby('image_id') #.count().reset_index(drop=False)
# # print('[df_results2]\n', df_results2)
# # df_results3 = df_results2.count()
# # print('[df_results3]\n', df_results3)
# # # exit()
# # df_results2 = df_results2.rename(columns={'image_id': 'id', 'class_label': 'count'})
# # df_results2.loc[df_statistics['class_label'] == 'Artifact', ''] = ''
# # df_results2['id'] = df_results2['id'].apply(lambda x: int(x[5:-4]))
# # df_results2 = df_results2.sort_values(by='id')
# # print('[df_results2]\n', df_results2)
# # df_results2.to_csv(os.path.join(base_path, 'predictions.csv'), index=False)
