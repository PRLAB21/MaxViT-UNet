MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'lysto'
MODEL_NAME = 'upernet_maxvit_timm'
PATH_DATASET = f'{DATASET_HOME_PATH}/{DATASET}'
CONFIG_FILE_NAME = MODEL_NAME + '_s' + str(S)
PATH_CONFIG_FILE = f'{MMSEG_HOME_PATH}/configs/{DATASET}/{CONFIG_FILE_NAME}.py'
PATH_WORK_DIR = f'{MMSEG_HOME_PATH}/trained_models/{DATASET}/{MODEL_NAME}/setting{S}/'

# The new config inherits a base config to highlight the necessary modification
_base_ = '../maxvit_timm/upernet_maxvit_timm-b16_ln_mln_512x512_160k_ade20k.py'

model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
    auxiliary_head=dict(
        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)])
    )

custom_imports = dict(
    imports=[
        # 'mmseg.models.backbones.lympnet2',
        'mmseg.core.utils.lymph_count_eval_hook',
    ],
    allow_failed_imports=False)

# Modify dataset related settings
dataset_type = 'LystoDataset'
data_root = PATH_DATASET
classes = ('background', 'lymphocyte')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
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
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        img_dir='val_IHC',
        ann_dir='val_circular_masks_label',
        split='mmseg_splits/val_circular_masks.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    # dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        img_dir='test_IHC',
        ann_dir='test_circular_masks_label',
        split='mmseg_splits/test_circular_masks.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(256, 256),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                    # dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

custom_hooks = [
    dict(
        type='LymphCountEvalHook',
        eval_interval=1,
        file_prefix=CONFIG_FILE_NAME,
        path_config=PATH_CONFIG_FILE,
        base_dir='eval1_circular_masks_label')
]

work_dir = PATH_WORK_DIR
# load_from = f'{MMSEG_HOME_PATH}/checkpoints/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth'
# load_from = os.path.join(PATH_WORK_DIR, 'latest.pth')
# resume_from = os.path.join(PATH_WORK_DIR, 'latest.pth')

total_epochs = max_epochs = 30
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=max_epochs)
# optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer = dict(_delete_=True, type='AdaBelief', lr=1e-3, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-2, weight_decouple=True, rectify=False)
# lr_config = dict(_delete_=True, policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.0025, step=[12, 20, 28])
lr_config = dict(by_epoch=True)
evaluation = dict(interval=1, metric=['mIoU', 'mDice'], pre_eval=True, save_best='mDice')
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=1, by_epoch=True)
seed = 0
gpu_ids = range(1)
workflow = [('train', 1), ('val', 1)]
