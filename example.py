norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MaxVitTimm', arch='maxvit_rmlp_nano_rw_256', pretrained=True),
    neck=dict(
        type='MultiLevelNeck',
        in_channels=[64, 128, 256, 512],
        out_channels=512,
        scales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[512, 512, 512, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ],
        output_channels=1),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ],
        output_channels=1),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'LystoDataset'
data_root = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
        type='LystoDataset',
        data_root='/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto',
        img_dir='train_IHC',
        ann_dir='train_circular_masks_label',
        split='mmseg_splits/train_circular_masks.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='LystoDataset',
        data_root='/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto',
        img_dir='val_IHC',
        ann_dir='val_circular_masks_label',
        split='mmseg_splits/val_circular_masks.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
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
        ]),
    test=dict(
        type='LystoDataset',
        data_root='/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto',
        img_dir='test_IHC',
        ann_dir='test_circular_masks_label',
        split='mmseg_splits/test_circular_masks.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
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
optimizer = dict(
    type='AdaBelief',
    lr=0.001,
    eps=1e-08,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    weight_decouple=True,
    rectify=False)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(by_epoch=True, interval=1)
evaluation = dict(
    interval=2, metric=['mIoU', 'mDice'], pre_eval=True, save_best=True)
MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'lysto'
MODEL_NAME = 'upernet_maxvit_timm'
PATH_DATASET = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto'
CONFIG_FILE_NAME = 'upernet_maxvit_timm_s1'
PATH_CONFIG_FILE = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/lysto/upernet_maxvit_timm_s1.py'
PATH_WORK_DIR = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/lysto/upernet_maxvit_timm/setting1/'
custom_imports = dict(
    imports=['mmseg.core.utils.lymph_count_eval_hook'],
    allow_failed_imports=False)
classes = ('background', 'lymphocyte')
custom_hooks = [
    dict(
        type='LymphCountEvalHook',
        eval_interval=2,
        file_prefix='upernet_maxvit_timm_s1',
        path_config=
        '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/lysto/upernet_maxvit_timm_s1.py',
        base_dir='eval1_circular_masks_label')
]
work_dir = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/lysto/upernet_maxvit_timm/setting1/'
total_epochs = 50
max_epochs = 50
seed = 0
gpu_ids = range(0, 1)
