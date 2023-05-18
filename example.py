norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MaxViT',
        in_channels=3,
        depths=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        embed_dim=64,
        num_heads=32,
        grid_window_size=(8, 8),
        attn_drop=0.1,
        drop=0.1,
        drop_path=0.1,
        mlp_ratio=4),
    decode_head=dict(
        type='MaxViTDecoder',
        in_channels=[64, 128, 256, 512],
        output_size=(256, 256),
        num_heads=32,
        grid_window_size=(8, 8),
        attn_drop=0.1,
        drop=0.1,
        drop_path=0.1,
        dropout_ratio=0.1,
        mlp_ratio=4.0,
        channels=64,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=5.0,
                ignore_index=0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=5.0,
                ignore_index=0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'MoNuSegDataset'
data_root = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/MoNuSeg'
img_norm_cfg = dict(
    mean=[171.3095, 119.6935, 157.7024],
    std=[56.044, 59.6094, 47.6912],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MoNuSegDataset',
        data_root=
        '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/MoNuSeg/raw/train',
        img_dir='imgs',
        ann_dir='masks_pngs_label',
        split='mmseg_splits.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomAffine'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[171.3095, 119.6935, 157.7024],
                std=[56.044, 59.6094, 47.6912],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='MoNuSegDataset',
        data_root=
        '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/MoNuSeg/raw/test',
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
                    dict(
                        type='Normalize',
                        mean=[171.3095, 119.6935, 157.7024],
                        std=[56.044, 59.6094, 47.6912],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='MoNuSegDataset',
        data_root=
        '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/MoNuSeg/raw/test',
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
                    dict(
                        type='Normalize',
                        mean=[171.3095, 119.6935, 157.7024],
                        std=[56.044, 59.6094, 47.6912],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(type='AdamW', lr=0.005, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.1,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=10)
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mDice', 'mFscore'],
    pre_eval=True,
    save_best='mDice')
MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'MoNuSeg'
MODEL_NAME = 'maxvit_unet'
PATH_DATASET = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/MoNuSeg'
CONFIG_FILE_NAME = 'maxvit_unet_s1'
PATH_CONFIG_FILE = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/MoNuSeg/maxvit_unet_s1.py'
PATH_WORK_DIR = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/MoNuSeg/maxvit_unet/setting1/'
classes = ('background', 'nuclei')
work_dir = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/MoNuSeg/maxvit_unet/setting1/'
total_epochs = 50
max_epochs = 50
seed = 0
gpu_ids = range(0, 1)
