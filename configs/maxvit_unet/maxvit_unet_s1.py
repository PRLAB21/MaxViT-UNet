MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'MoNuSeg'
MODEL_NAME = 'maxvit_unet'
PATH_DATASET = f'{DATASET_HOME_PATH}/{DATASET}'
CONFIG_FILE_NAME = MODEL_NAME + '_s' + str(S)
PATH_CONFIG_FILE = f'{MMSEG_HOME_PATH}/configs/{DATASET}/{CONFIG_FILE_NAME}.py'
PATH_WORK_DIR = f'{MMSEG_HOME_PATH}/trained_models/{DATASET}/{MODEL_NAME}/setting{S}/'

# The new config inherits a base config to highlight the necessary modification
_base_ = [
    '../_base_/models/maxvit_unet.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        attn_drop=0.10,
        drop=0.10,
        drop_path=0.10),
    decode_head=dict(
        num_classes=2,
        attn_drop=0.10,
        drop=0.10,
        drop_path=0.10,
        dropout_ratio=0.10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=5.0, ignore_index=0),
        ]),
    auxiliary_head=dict(
        num_classes=2,
        dropout_ratio=0.10,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=5.0, ignore_index=0),
        ]),
    )

# Modify dataset related settings
dataset_type = 'MoNuSegDataset'
data_root = PATH_DATASET
classes = ('background', 'nuclei')
# Normalization parameters specific to MoNuSeg18 dataset
img_norm_cfg = dict(mean=[171.3095, 119.6935, 157.7024], std=[56.0440, 59.6094, 47.6912], to_rgb=True)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=f'{data_root}/raw/train',
        # full
        img_dir='imgs',
        ann_dir='masks_pngs_label',
        split='mmseg_splits.txt',
        # subset
        # img_dir='imgs_subset',
        # ann_dir='masks_pngs_subset_label',
        # split='mmseg_splits_subset.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomAffine'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        _delete_=True,
        type=dataset_type,
        data_root=f'{data_root}/raw/test',
        img_dir='imgs',
        ann_dir='masks_pngs_label',
        split='mmseg_splits.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        _delete_=True,
        type=dataset_type,
        data_root=f'{data_root}/raw/test',
        img_dir='imgs',
        ann_dir='masks_pngs_label',
        split='mmseg_splits.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

work_dir = PATH_WORK_DIR
# load_from = f'{MMSEG_HOME_PATH}/checkpoints/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth'
# resume_from = f'{PATH_WORK_DIR}/latest.pth'

total_epochs = max_epochs = 50
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=max_epochs)
optimizer = dict(_delete_=True, type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.01)
lr_config = dict(_delete_=True, policy='CosineAnnealing', warmup='linear', warmup_iters=100, warmup_ratio=0.1, min_lr_ratio=0.001)
evaluation = dict(interval=1, metric=['mIoU', 'mDice', 'mFscore'], pre_eval=True, save_best='mDice')
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=total_epochs//5, by_epoch=True)
dist_params = dict(backend='nccl')
cudnn_benchmark = True
seed = 0
gpu_ids = range(1)
workflow = [('train', 1), ('val', 1)]
