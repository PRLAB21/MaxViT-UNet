norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='EfficientUNet', model_name='efficientnet-b5'),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0)
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
dataset_type = 'LystoDataset'
data_root = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (2336, 3504)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2336, 3504), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2336, 3504),
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
        ann_dir='train_mask3_label',
        split='mmseg_splits/train_mask3.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
        ann_dir='val_mask3_label',
        split='mmseg_splits/val_mask3.txt',
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
        ann_dir='test_mask3_label',
        split='mmseg_splits/test_mask3.txt',
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
    type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(by_epoch=True, interval=2)
evaluation = dict(interval=2, metric=['mIoU', 'mDice'], pre_eval=True)
MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'lysto'
MODEL_NAME = 'efficient_unet_b5'
PATH_DATASET = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto'
CONFIG_FILE_NAME = 'efficient_unet_b5_s1'
PATH_CONFIG_FILE = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/lysto/efficient_unet_b5_s1.py'
PATH_WORK_DIR = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/lysto/efficient_unet_b5/setting1/'
custom_imports = dict(
    imports=[
        'mmseg.core.utils.lymph_count_eval_hook',
        'mmseg.models.backbones.efficientunet'
    ],
    allow_failed_imports=False)
classes = ('background', 'lymphocyte')
custom_hooks = [
    dict(
        type='LymphCountEvalHook',
        eval_interval=2,
        file_prefix='efficient_unet_b5',
        path_config=
        '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/lysto/efficient_unet_b5_s1.py',
        base_dir='eval1_mask3_label')
]
work_dir = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/trained_models/lysto/efficient_unet_b5/setting1/'
total_epochs = 30
max_epochs = 30
seed = 0
gpu_ids = range(0, 1)
