import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MMSEG_HOME_PATH = '../../'

def old_code():
    labels = recall_precision_data.keys()
    print(TAG, '[labels]', labels)
    x = np.arange(len(labels))  # the label locations
    width = 0.6  # the width of the bars

    # create figure object
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    i = 1
    while os.path.exists(f'trained_models/lysto-models/z-recall_precision_curves/{i:03}-recall-plot.jpg'):
        i += 1

    for name, y in [['recall', recalls], ['precision', precisions]]:
        fig.suptitle(f'{name.capitalize()} Comparison @ {threshold_type}', fontsize=20)
        rects = ax.bar(x + width/2, y, width, color=[[0, 1, 1, 0.5], [1, 0, 1, 0.5], [1, 1, 0, 0.5], [0, 0, 1, 0.5], [1, 0, 0, 1]])
        ax.set_ylabel(name.capitalize(), fontsize=18)
        ax.set_xticks(x, labels, rotation=30, fontsize=18)
        ax.bar_label(rects, padding=10, fmt='%.3f', fontsize=18)
        ax.set_ylim([0, 1])
        plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
        print(f'trained_models/lysto-models/z-recall_precision_curves/{i:03}-{name}-plot.jpg')
        plt.savefig(opj(f'trained_models/lysto-models/z-recall_precision_curves/{i:03}-{name}-plot.jpg'), dpi=300)
        ax.cla()

    # create figure object
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes = axes.ravel()
    fig.set_size_inches((20, 10))
    # set figure title
    fig.suptitle(f'F-Score Curves Comparison @ {threshold_type}', fontsize=18)
    # draw bar plots
    rects1 = axes[0].bar(x + width/2, recalls, width, color=[[0, 1, 1, 0.5], [1, 0, 1, 0.5], [1, 1, 0, 0.5], [0, 0, 1, 0.5], [1, 0, 0, 1]])
    rects2 = axes[1].bar(x + width/2, precisions, width, color=[[0, 1, 1, 0.5], [1, 0, 1, 0.5], [1, 1, 0, 0.5], [0, 0, 1, 0.5], [1, 0, 0, 1]])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axes[0].set_ylabel('Recall', fontsize=14)
    axes[0].set_xticks(x, labels, rotation=30, fontsize=14)
    axes[0].bar_label(rects1, padding=10, fmt='%.3f', fontsize=14)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axes[1].set_ylabel('Precision', fontsize=14)
    axes[1].set_xticks(x, labels, rotation=30, fontsize=14)
    axes[1].bar_label(rects2, padding=10, fmt='%.3f', fontsize=14)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    i = 1
    while os.path.exists(f'trained_models/lysto-models/z-recall_precision_curves/recall-precision-curve-{i:03}.jpg'):
        i += 1
    print(f'trained_models/lysto-models/z-recall_precision_curves/recall-precision-curve-{i:03}.jpg')
    plt.savefig(opj(f'trained_models/lysto-models/z-recall_precision_curves/recall-precision-curve-{i:03}.jpg'), dpi=300)
    plt.show()

def plot_metric_all_thresholds(metric_name, proposed_model_name, plot_save_path, min_epoch, max_epoch):
    """ This function plots subplots for `metric_name` at all `threshold_types` """
    # proposed_model_name = 'PVTCB-Lymph-Det'
    TAG2 = TAG + '[plot_metric_all_thresholds]'
    if not os.path.exists(plot_save_path):
        print(TAG2, 'creating path:', plot_save_path)
        os.mkdir(plot_save_path)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    fig.suptitle(f'{metric_name} comparison at various thresholds', fontsize=18)
    axes = axes.ravel()
    
    for label, eval_csv_path in eval_csv_paths.items():
        # print(TAG, '[label, eval_csv_path]', label, eval_csv_path)
        df_eval = pd.read_csv(eval_csv_path)
        df_eval = df_eval.sort_values(by='epoch')

        for i, threshold_type in enumerate(threshold_types):
            # print(threshold_type, df_eval.threshold == threshold_type)
            condition = (df_eval.distance == 12) & (df_eval.threshold == threshold_type) & (df_eval.epoch >= min_epoch) & (df_eval.epoch <= max_epoch)
            epochs = df_eval.loc[condition, 'epoch'].values
            epochs = epochs[1:]

            if threshold_type == 'names':
                metric_values = np.zeros_like(epochs)
            else:
                metric_values = df_eval.loc[condition, metric_name]
                metric_values = metric_values[1:]

            # idx_max_fscore = np.argmax(metric_values)
            # print(TAG, label, threshold_type, '[idx_max_fscore]', idx_max_fscore, '\n', df_eval.iloc[idx_max_fscore])

            if proposed_model_name in label:
                # axes[i].plot(epochs, metric_values, color='r', linewidth=3, label=label)
                axes[i].plot(epochs, metric_values, '-r', label=label, linewidth=3)
            else:
                # axes[i].plot(epochs, metric_values, linewidth=2, label=label)
                axes[i].plot(epochs, metric_values, '--', label=label, linewidth=2)
            
            axes[i].set_title(threshold_type, fontsize=14)
            # axes[i].set_xticks(epochs)
            # axes[i].set_xlim((0, np.max(epochs)))
            # axes[i].set_ylim((0, 1))
            axes[i].set_xlabel('Epochs', fontsize=14)
            axes[i].set_ylabel(metric_name.capitalize(), fontsize=14)
            # plt.legend(fontsize=14)

    axes[-1].legend(fontsize=12)
    plt.tight_layout()
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    i = 1
    while os.path.exists(opj(plot_save_path, f'eval_metrices-{metric_name}-{i:03}.jpg')):
        i += 1
    plot_path = opj(plot_save_path, f'eval_metrices-{metric_name}-{i:03}.jpg')
    print(TAG2, 'plot saved at:', plot_path)
    plt.savefig(plot_path, dpi=300)
    plt.show()

def plot_metrices_at_threshold(plot_threshold, proposed_model_name, plot_save_path, min_epoch, max_epoch):
    """ this function plots subplots for all metrices at particular threshold (should be one of threshold_types) """
    TAG2 = TAG + '[plot_metrices_at_threshold]'
    if not os.path.exists(plot_save_path):
        print(TAG2, 'creating path:', plot_save_path)
        os.mkdir(plot_save_path)

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
    fig.suptitle(f'Metrices comparison at {plot_threshold}', fontsize=18)
    axes = axes.ravel()

    for label, eval_csv_path in eval_csv_paths.items():
        df_eval = pd.read_csv(eval_csv_path)
        df_eval = df_eval.sort_values(by='epoch')
        condition = (df_eval.distance == 12) & (df_eval.threshold == plot_threshold) & (df_eval.epoch >= min_epoch) & (df_eval.epoch <= max_epoch)
        epochs = df_eval.loc[condition, 'epoch'].values
        # epochs = epochs[:max_epoch]

        for i, metric in enumerate(metrices):
            if metric == 'names':
                epochs = 0
                metric_values = 0
            else:
                metric_values = df_eval.loc[condition, metric].values
                # metric_values = metric_values[:max_epoch]

            if proposed_model_name in label:
                # axes[i].plot(epochs, metric_values, color='r', linewidth=3, label=label)
                axes[i].plot(epochs, metric_values, '-r', label=label, linewidth=3)
            else:
                # axes[i].plot(epochs, metric_values, linewidth=2, label=label)
                axes[i].plot(epochs, metric_values, '--', label=label, linewidth=2)

            axes[i].set_title(metric.capitalize())
            # axes[i].set_xticks(epochs)
            # axes[i].set_ylim((0, 1))
            axes[i].set_xlabel('Epochs', fontsize=14)
            axes[i].set_ylabel(metric.capitalize(), fontsize=14)

        # condition = (df_eval.distance == 12) & (df_eval.threshold == plot_threshold) & (df_eval.epoch == last_epoch)
        # df_last_epoch = df_eval.loc[condition]
        # df_last_epoch['model'] = label
        # metrics_combined.append(df_last_epoch)
        # print(TAG, label)
        # print(df_eval.loc[condition])

    # metrics_combined = pd.concat(metrics_combined, ignore_index=True)
    # print(TAG, '[metrics_combined]\n', metrics_combined)

    axes[-1].legend(fontsize=12)
    plt.tight_layout()
    # plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    i = 1
    while os.path.exists(opj(plot_save_path, f'eval_metrices-{plot_threshold}-{i:03}.jpg')):
        i += 1
    plot_path = opj(plot_save_path, f'eval_metrices-{plot_threshold}-{i:03}.jpg')
    print(TAG2, 'plot saved at:', plot_path)
    plt.savefig(plot_path, dpi=300)
    plt.show()

TAG = '[z-plot_eval_metrices]'
opj = os.path.join
lysto_path = opj(MMSEG_HOME_PATH, 'trained_models/lysto-models/')
lyon_path = opj(MMSEG_HOME_PATH, 'trained_models/lyon-models/')
eval_csv_paths = {
    # 'maskrcnn-lymphnet3-cm1-s4-1-e1': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting4-2022-03-13/evaluation1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphnet3-cm1-s4-1-e2': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting4-2022-03-13/evaluation2/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphnet3-cm1-s4': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting4/eval1_mask_images1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphnet3-cm1-s5': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting5/eval1_mask_images1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphnet3-cm1-s6': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting6/eval1_mask_images1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'CB-RCNN': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting7/eval1_mask_images1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'Maskrcnn Resnet50': opj(lysto_path, 'maskrcnn-resnet50/setting1/eval1_mask_images1/maskrcnn-resnet50-test.csv'),
    # 'Maskrcnn Resnet CBAM50': opj(lysto_path, 'maskrcnn-resnet-cbam50/setting1/eval1_mask_images1/maskrcnn-resnet-cbam50-test.csv'),
    # 'Maskrcnn ResNeXt50': opj(lysto_path, 'maskrcnn-resnext50/setting1/eval1_mask_images1/maskrcnn-resnext50-test.csv'),
    # 'Maskrcnn Resnet50 (Dilated Conv)': opj(lysto_path, 'maskrcnn-resnet50-dilation1223/setting1/eval1_mask_images1/maskrcnn-resnet50-dilation1223-test.csv'),
    # 'Cascade Maskrcnn Resnet50': opj(lysto_path, 'cascade-maskrcnn-resnet50/setting1/eval1_mask_images1/cascade-maskrcnn-resnet50-test.csv'),
    # 'PVTCB-Lymph-Det': opj(lysto_path, 'maskrcnn-lymphocytenet-pvt/setting2/eval1_mask_images1/maskrcnn-lymphocytenet-pvt-test.csv'),

    # 'retinanet-resnet50-s1': opj(lyon_path, 'retinanet-resnet50/setting1/eval1_mask_images1/retinanet-resnet50-test.csv'),
    # 'fasterrcnn-resnet50-s1': opj(lyon_path, 'fasterrcnn-resnet50/setting1/eval1_mask_images1/fasterrcnn-resnet50-test.csv'),

    # 'maskrcnn-lymphnet3-cm1-s11-avg': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting11/eval1_mask3-maskrcnn-lymphocytenet3-cm1-s11-val.csv'),
    # 'maskrcnn-lymphnet3-cm1-s12-avg': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting12/eval1_mask3-maskrcnn-lymphocytenet3-cm1-s12-val.csv'),
    # 'maskrcnn-lymphnet3-cm1-s13-avg': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting13/eval1_mask3-maskrcnn-lymphocytenet3-cm1-s13-val.csv'),
    # 'maskrcnn-lymphnet3-cm1-s14-avg': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting14/eval1_mask3-maskrcnn-lymphocytenet3-cm1-s14-val.csv'),
    # 'maskrcnn-lymphnet3-cm1-s15-avg': opj(lysto_path, 'maskrcnn-lymphocytenet3-cm1/setting15/eval1_mask3-maskrcnn-lymphocytenet3-cm1-s15-val.csv'),
    
    # 'YOLOX-S (Pre-Trained)': opj(lyon_path, 'yolox-s/setting1/eval1/yolox-s-test.csv'),
    # 'YOLOX-S (Scratch)': opj(lyon_path, 'yolox-s/setting2/eval1/yolox-s-test.csv'),
    # 'RetinaNet ResNet50 (Pre-Trained)': opj(lyon_path, 'retinanet-resnet50/setting1/eval1/retinanet-resnet50-test.csv'),
    # 'SCNet ResNet50 (Pre-Trained)': opj(lyon_path, 'scnet-resnet50/setting1/eval1/scnet-resnet50-test.csv'),
    # 'FasterRCNN ResNet50 (Pre-Trained)': opj(lyon_path, 'fasterrcnn-resnet50/setting1/eval1/fasterrcnn-resnet50-test.csv'),
    'MaskRCNN ResNet50 (s1 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting1/eval1/maskrcnn-resnet50-test.csv'),
    'MaskRCNN ResNet50 (s2 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting2/eval1/maskrcnn-resnet50-test.csv'),
    'MaskRCNN ResNet50 (s3-1 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting3-1/eval1/maskrcnn-resnet50-test.csv'),
    'MaskRCNN ResNet50 (s3 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting3/eval1/maskrcnn-resnet50-test.csv'),
    'MaskRCNN ResNet50 (s4 Pre-Trained)': opj(lyon_path, 'maskrcnn-resnet50/setting4/eval1/maskrcnn-resnet50-test.csv'),
    # 'MaskRCNN LymphocyteNet3 (Proposed)': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting6/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s7': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting7/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s13-dab': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting13_DAB/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s13-hsv': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting13_HSV/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s13': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting13/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
    # 'maskrcnn-lymphocytenet3-cm1-s14': opj(lyon_path, 'maskrcnn-lymphocytenet3-cm1/setting14/eval1/maskrcnn-lymphocytenet3-cm1-test.csv'),
}

threshold_types = ['threshold=0.25', 'threshold=0.50', 'threshold=0.75', 'threshold=0.95', 'threshold=0.5:0.95', 'names']
metrices = ['recall', 'precision', 'fscore', 'kappa', 'accuracy', 'names']
proposed_model_name = 'MaskRCNN LymphocyteNet3'
proposed_model_name = 's4'
plot_metrices_save_path = opj(MMSEG_HOME_PATH, 'trained_models/lyon-models/z-plot_metrices_at_threshold')
plot_thresholds_save_path = opj(MMSEG_HOME_PATH, 'trained_models/lyon-models/z-plot_metric_all_thresholds')
plot_metrices_at_threshold('threshold=0.50', proposed_model_name, plot_metrices_save_path, min_epoch=1, max_epoch=16)
plot_metric_all_thresholds('fscore', proposed_model_name, plot_thresholds_save_path, min_epoch=1, max_epoch=16)
