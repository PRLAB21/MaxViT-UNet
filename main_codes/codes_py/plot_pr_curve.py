import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

MMSEG_HOME_PATH = '../../'

def make_monotonic(array):
    monotonic_array = []
    previous_largest = -1
    for i, value in enumerate(array):
        if value > previous_largest:
            previous_largest = value
        monotonic_array.append(previous_largest)
    return monotonic_array

def get_values(df_stats, thresholds):
    TPs, FPs, FNs = [], [], []
    recalls, precisions, fscores = [], [], []

    # calculate pr-curve values for one csv file data
    for threshold in thresholds:
        # select rows for this iteration threshold
        condition = (df_stats.distance == d) & (df_stats.threshold == threshold)
        # accumulate TP, FP, FN related to selected rows
        # print(TAG2, threshold, df_stats)
        TP = df_stats.loc[condition, 'TP'].sum()
        FP = df_stats.loc[condition, 'FP'].sum()
        FN = df_stats.loc[condition, 'FN'].sum()
        # append TP, FP, FN
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)
        # calculate recall, precision, fscore
        recall = TP / (TP + FN + 1e-10)
        precision = 1 if np.isnan(TP / (TP + FP)) else TP / (TP + FP)
        fscore = (2 * precision * recall) / (precision + recall)
        # append recall, precision, fscore
        recalls.append(recall)
        precisions.append(precision)
        fscores.append(fscore)

        # show some logs on console
        # if threshold == 0.5:
        #     df_summary['Model'].append(plot_label)
        #     df_summary['TP'].append(TP)
        #     df_summary['FP'].append(FP)
        #     df_summary['FN'].append(FN)
        #     df_summary['Recall'].append(recall)
        #     df_summary['Precision'].append(precision)
        #     df_summary['F1-Score'].append(fscore)

    # append extra values at both end for complete pr curve from 0 to 1
    thresholds.insert(0, -1)
    recalls.insert(0, 1)
    precisions.insert(0, 0)
    fscores.insert(0, -1)
    # print(pd.DataFrame({'thresholds': thresholds, 'recalls': recalls, 'precisions': precisions, 'fscores': fscores}).head(100))

    return TPs, FPs, FNs, recalls, precisions, fscores

def plot_from_pr_csv(proposed_model_name, plot_save_path, dataset_name):
    """ this function plots comparision PR-Curve using pr_curve csv calculated from inference.py script """
    TAG2 = TAG + '[plot_pr_curve_fron_stats]'
    if not os.path.exists(plot_save_path):
        print(TAG2, 'creating path:', plot_save_path)
        os.mkdir(plot_save_path)

    plt.figure(figsize=(10, 10))
    plt.suptitle(f'PR-Curve Comparision | {dataset_name} Dataset', fontsize=18)

    for label, csv_path in stats_csv_paths.items():
        df_pr_curve = pd.read_csv(csv_path)
        recalls = df_pr_curve['recalls'].values.tolist()
        precisions = df_pr_curve['precisions'].values.tolist()
        fscores = df_pr_curve['fscores'].values.tolist()

        # recalls.insert(0, 1)
        # precisions.insert(0, 0)
        # fscores.insert(0, -1)

        recalls, precisions = make_monotonic(recalls[::-1]), precisions[::-1]
        auc_pr_curve = auc(recalls, precisions)
        plot_label = f'{plot_label} (auc = {auc_pr_curve:0.4f})'
        if proposed_model_name in plot_label:
            plt.plot(recalls, precisions, '-r', label=plot_label, linewidth=3)
        else:
            plt.plot(recalls, precisions, '--', label=plot_label, linewidth=2)

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(fontsize=14)
    # plt.tight_layout()
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    i = 1
    while os.path.exists(opj(plot_save_path, f'pr-curve-{i:03}.jpg')):
        i += 1
    plot_path = opj(plot_save_path, f'pr-curve-{i:03}.jpg')
    print(TAG2, 'plot saved at:', plot_path)
    plt.savefig(plot_path, dpi=300)
    plt.show()

def plot_pr_curve_fron_stats(proposed_model_name, plot_save_path, dataset_name):
    """ this function plots comparision PR-Curve using statistics calculated from inference.py script """
    TAG2 = TAG + '[plot_pr_curve_fron_stats]'
    if not os.path.exists(plot_save_path):
        print(TAG2, 'creating path:', plot_save_path)
        os.mkdir(plot_save_path)

    plt.figure(figsize=(10, 10))
    plt.suptitle(f'PR-Curve Comparision | {dataset_name} Dataset', fontsize=18)

    # df_summary = {'Model': [], 'TP': [], 'FP': [], 'FN': [], 'Recall': [], 'Precision': [], 'F1-Score': []}
    for label, csv_path in stats_csv_paths.items():
        print(TAG, '[label, csv_path]', label, csv_path)
        df_stats = pd.read_csv(csv_path)
        thresholds = pd.unique(df_stats.threshold).tolist()
        # print(TAG, label, '[thresholds]', thresholds)
        TPs, FPs, FNs, recalls, precisions, fscores = get_values(df_stats, thresholds)
        
        # make_monotonic will adjust recall, precision values a bit for auc calculation
        recalls2, precisions2 = make_monotonic(recalls[::-1]), precisions[::-1]
        auc_pr_curve = auc(recalls2, precisions2)
        
        # idx_max_fscore = np.array(fscores).argmax()
        # print(f'{plot_label}: max fscore is {fscores[idx_max_fscore]:.4f} at threshold {thresholds[idx_max_fscore]}')
        # print(f'{plot_label}: TP: {np.mean(TPs):.4f}')
        # print(f'{plot_label}: FP: {np.mean(FPs):.4f}')
        # print(f'{plot_label}: FN: {np.mean(FNs):.4f}')
        # print(f'{plot_label}: mean recall: {np.mean(recalls):.4f}')
        # print(f'{plot_label}: mean precision: {np.mean(precisions):.4f}')
        # print(f'{plot_label}: mean fscore: {np.mean(fscores):.4f}')
        
        label = f'{label} (auc = {auc_pr_curve:0.4f})'
        if proposed_model_name in label:
            plt.plot(recalls2, precisions2, '-r', label=label, linewidth=3)
        else:
            plt.plot(recalls2, precisions2, '--', label=label)

    # pd.DataFrame(df_summary).to_csv(opj(base_path, f'comparision-summary-{dataset_name}-dataset-01.csv'))

    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(fontsize=14)
    # plt.tight_layout()
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    i = 1
    while os.path.exists(opj(plot_save_path, f'pr-curve-{i:03}.jpg')):
        i += 1
    plot_path = opj(plot_save_path, f'pr-curve-{i:03}.jpg')
    print(TAG2, 'plot saved at:', plot_path)
    plt.savefig(plot_path, dpi=300)
    plt.show()

TAG = '[z-plot_pr_curve]'
opj = os.path.join
lysto_path = opj(MMSEG_HOME_PATH, 'trained_models/lysto-models/')
lyon_path = opj(MMSEG_HOME_PATH, 'trained_models/lyon-models/')
stats_csv_paths = {
    # '': opj(lyon_path, 'work-dir-cascade-maskrcnn-resnet50/setting1/images-stats-per-threshold-lyon-cascade-maskrcnn-resnet50.csv'),
    # '': opj(lyon_path, 'work-dir-maskrcnn-resnet50/setting3/images-stats-per-threshold-lyon-maskrcnn-resnet50.csv'),
    # '': opj(lyon_path, 'work-dir-maskrcnn-resnetcbam50/setting8/images-stats-per-threshold-lyon-maskrcnn-resnetcbam50.csv'),
    # '': opj(lyon_path, 'work-dir-maskrcnn-resnext101/setting1/images-stats-per-threshold-lyon-maskrcnn-resnext101.csv'),
    # '': opj(lyon_path, 'work-dir-scnet-resnet50/setting1/images-stats-per-threshold-lyon-scnet-resnet50.csv'),
    # '': opj(lyon_path, 'work-dir-maskrcnn-lymphocytenet3-cm1/setting4-trained-on-lysto/images-stats-per-threshold-lyon-maskrcnn-lymphocytenet3-cm1.csv'),
    # 'Cascade MaskRCNN Resnet50': opj(lysto_path, 'cascade-maskrcnn-resnet50/setting1/statistics-cascade-maskrcnn-resnet50-s1.csv'),
    # 'MaskRCNN ResNeXt50': opj(lysto_path, 'maskrcnn-resnext50/setting1/statistics-maskrcnn-resnext50-s1.csv'),
    # 'MaskRCNN Resnet50 (Dilated Conv)': opj(lysto_path, 'maskrcnn-resnet50-dilation1223/setting1/statistics-maskrcnn-resnet50-dilation1223-s1.csv'),
    # 'MaskRCNN ResNetCBAM50': opj(lysto_path, 'maskrcnn-resnet-cbam50/setting1/statistics-maskrcnn-resnet-cbam50-s1.csv'),
    # 'MaskRCNN Resnet50': opj(lysto_path, 'maskrcnn-resnet50/setting1/statistics-maskrcnn-resnet50-s1.csv'),
    # 'MaskRCNN Lymphocyte s4': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting4/statistics-maskrcnn-lymphocytenet3-cm1-s4.csv'),
    # 'MaskRCNN Lymphocyte s5': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting5/statistics-maskrcnn-lymphocytenet3-cm1-s5.csv'),
    # 'MaskRCNN Lymphocyte s6 ep30': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting6/statistics-maskrcnn-lymphocytenet3-cm1-s6-ep30.csv'),
    # 'MaskRCNN Lymphocyte s7 ep30': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting7/statistics-maskrcnn-lymphocytenet3-cm1-s7-ep30.csv'),
    # 'MaskRCNN ResNet50': opj(lysto_path, 'maskrcnn-resnet50/setting2/statistics-maskrcnn-resnet50-s2-ep30.csv'),
    # 'SCNet ResNet50': opj(lysto_path, 'scnet-resnet50/setting1/statistics-scnet-resnet50-s1-ep30.csv'),
    # 'PVTCB-Lymph-Det': opj(lysto_path, 'maskrcnn-lymphocytenet-pvt/setting2/statistics-maskrcnn-lymphocytenet-pvt-s2-ep30.csv'),
    # 'Attention-3x3': opj(lysto_path, 'maskrcnn-attention_3x3/setting1/statistics-maskrcnn-attention_3x3-s1-ep30.csv'),

    # 'YOLOX-S (Pre-Trained)': opj(lyon_path, 'yolox-s/setting1/infer_lyon_testset/statistics-yolox-s-s1-ep15.csv'),
    # 'YOLOX-S (Scratch)': opj(lyon_path, 'yolox-s/setting2/infer_lyon_testset/statistics-yolox-s-s2-ep15.csv'),
    # 'RetinaNet ResNet50 (Pre-Trained)': opj(lyon_path, 'retinanet-resnet50/setting1/infer_lyon_testset/statistics-retinanet-resnet50-s1-ep15.csv'),
    # 'SCNet ResNet50 (Pre-Trained)': opj(lyon_path, 'scnet-resnet50/setting1/infer_lyon_testset/statistics-scnet-resnet50-s1-ep15.csv'),
    # 'FasterRCNN ResNet50 (Pre-Trained)': opj(lyon_path, 'fasterrcnn-resnet50/setting1/infer_lyon_testset/statistics-fasterrcnn-resnet50-s1-ep15.csv'),
    'MaskRCNN ResNet50 (s1 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting1/infer_lyon_testset/statistics-maskrcnn-resnet50-s1-ep15.csv'),
    'MaskRCNN ResNet50 (s2 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting2/infer_lyon_testset/statistics-maskrcnn-resnet50-s2-ep15.csv'),
    'MaskRCNN ResNet50 (s3-1 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting3-1/infer_lyon_testset/statistics-maskrcnn-resnet50-s3-ep15.csv'),
    'MaskRCNN ResNet50 (s3 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting3/infer_lyon_testset/statistics-maskrcnn-resnet50-s3-ep15.csv'),
    'MaskRCNN ResNet50 (s4 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting4/infer_lyon_testset/statistics-maskrcnn-resnet50-s4-ep15.csv'),
    # 'MaskRCNN LymphocyteNet3 (Proposed e10)': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting6/infer_lyon_testset/statistics-maskrcnn-lymphocytenet3-cm1-s6-ep10.csv'),
    # 'MaskRCNN LymphocyteNet3 (Proposed e20)': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting6/infer_lyon_testset/statistics-maskrcnn-lymphocytenet3-cm1-s6-ep20.csv'),
    # 'MaskRCNN LymphocyteNet3 (Proposed e30)': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting6/infer_lyon_testset/statistics-maskrcnn-lymphocytenet3-cm1-s6-ep30.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s14': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting14/inference_lyon/statistics-maskrcnn-lymphocytenet3-cm1-s14.csv'),
}

d = 12
# proposed_model_name = 'DAttn CB Lym Net'
# proposed_model_name = 'PVTCB-Lymph-Det'
proposed_model_name = 'MaskRCNN LymphocyteNet3'
proposed_model_name = 's3'
plot_pr_save_path = opj(MMSEG_HOME_PATH, 'trained_models/lyon-models/z-plot_pr_curve')
plot_pr_curve_fron_stats(proposed_model_name, plot_pr_save_path, 'Lyon')
