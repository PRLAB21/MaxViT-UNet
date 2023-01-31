import os
import pandas as pd

def compare_by_best_metric_value(metric_name, threshold_type, d, csv_save_path, file_count=1):
    """ This function compare the models based on the best value for metric_name specified. """
    TAG2 = TAG + '[compare_by_best_metric_value]'
    if not os.path.exists(csv_save_path):
        print(TAG2, 'creating path:', csv_save_path)
        os.mkdir(csv_save_path)

    column_order = ['model_config', 'epoch', 'distance', 'threshold', 'class_label', 'fscore', 'recall', 'precision', 'kappa', 'accuracy']
    df_best_metrics = pd.DataFrame()
    for key, value in eval_csv_paths.items():
        print(TAG2, key, value)
        df = pd.read_csv(value)
        df = df.sort_values(by='epoch')
        # for d in [0, 12]:
        #     for threshold_idx, threshold_type in enumerate(threshold_types):
        condition = (df.distance == d) & (df.threshold == threshold_type)
        idx_max_fscore = df.loc[condition, metric_name].idxmax()
        selected_row = df.loc[idx_max_fscore]
        selected_row['model_config'] = key
        df_best_metrics = df_best_metrics.append(selected_row)
    df_best_metrics = df_best_metrics[column_order]
    print(TAG2, '[df_best_metrics]\n', df_best_metrics)
    if threshold_types.index(threshold_type) == 4:
        threshold_str = '145'
    else:
        threshold_str = threshold_type.split('=')[1].replace('.', '')
    df_best_metrics.to_csv(os.path.join(csv_save_path, f'metric_{metric_name}-t_{threshold_str}-d_{d}-{file_count:03}.csv'), index=False)

def compare_folds_by_best_metric(path_work_dir, folds, eval_folder_name, metric_name, csv_save_path, file_count=1):
    df_combined = []
    # base_path = 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting15'
    for f in range(1, folds + 1):
        csv_path = os.path.join(path_work_dir, f'fold{f}', eval_folder_name, 'maskrcnn-lymphocytenet3-cm1-val.csv')
        print(csv_path, os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='epoch')
        df = df.loc[df.epoch.isin(list(range(2, 21, 2)))]
        df['fold'] = f
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

    # df_best_metrics = pd.DataFrame()
    # for d in [0, 12]:
    #     for threshold_idx, threshold_type in enumerate(threshold_types):
    #         condition = (df.distance == d) & (df.threshold == threshold_type)
    #         idx_max_fscore = df.loc[condition, 'fscore'].idxmax()
    #         selected_row = df.loc[idx_max_fscore]
    #         selected_row['model_config'] = key
    #         df_best_metrics = df_best_metrics.append(selected_row)

    # df_best_metrics = df_best_metrics[column_order]
    print(TAG, '[df_best_metrics]\n', df_best_metrics)
    # df_best_metrics.to_csv('z-get_best_metrics_fold_wise.csv', index=False)

TAG = '[z-compare_models]'
eval_csv_paths = {
    'maskrcnn-lymphnet3-cm1-s11-avg': 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting11/eval1_mask3/maskrcnn-lymphocytenet3-cm1-s11-val.csv',
    'maskrcnn-lymphnet3-cm1-s12-avg': 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting12/eval1_mask3/maskrcnn-lymphocytenet3-cm1-s12-val.csv',
    'maskrcnn-lymphnet3-cm1-s13-avg': 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting13/eval1_mask3/maskrcnn-lymphocytenet3-cm1-s13-val.csv',
    'maskrcnn-lymphnet3-cm1-s14-avg': 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting14/eval1_mask3/maskrcnn-lymphocytenet3-cm1-s14-val.csv',
    'maskrcnn-lymphnet3-cm1-s15-avg': 'trained_models/lysto-models/maskrcnn-lymphocytenet3-cm1/setting15/eval1_mask3/maskrcnn-lymphocytenet3-cm1-s15-val.csv',

    'retinanet-resnet50-s1': 'trained_models/lyon-models/retinanet-resnet50/setting1/eval1/retinanet-resnet50-test.csv',
    'fasterrcnn-resnet50-s1': 'trained_models/lyon-models/fasterrcnn-resnet50/setting1/eval1/fasterrcnn-resnet50-test.csv',
}
dataset_name = 'LYSTO'
threshold_types = ['threshold=0.25', 'threshold=0.50', 'threshold=0.75', 'threshold=0.95', 'threshold=0.5:0.95']
metrices = ['recall', 'precision', 'fscore', 'kappa', 'accuracy', 'names']
compare_save_path = 'trained_models/lyon-models/z-compare_by_best_metric_value'
compare_by_best_metric_value(metric_name=metrices[2], threshold_type=threshold_types[1], d=12)
