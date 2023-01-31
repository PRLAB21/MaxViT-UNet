MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'lysto'
MODEL_NAME = 'fcn_unet_s5'
PATH_DATASET = f'{DATASET_HOME_PATH}/{DATASET}'
CONFIG_FILE_NAME = MODEL_NAME + '_s' + str(S)
PATH_CONFIG_FILE = f'{MMSEG_HOME_PATH}/configs/{DATASET}_easy_vs_hard/{CONFIG_FILE_NAME}.py'
PATH_WORK_DIR = f'{MMSEG_HOME_PATH}/trained_models/lysto_easy/{MODEL_NAME}/setting{S}/'

# The new config inherits a base config to highlight the necessary modification
_base_ = '../unet/fcn_unet_s5-d16_256x256_40k_hrf.py'

# norm_cfg = dict(type='BN', requires_grad=True)
# model = dict(
#     type='EncoderDecoder',
#     backbone=dict(
#         type='UNet',
#         norm_cfg=dict(type='BN', requires_grad=True)),
#     decode_head=dict(
#         type='FCNHead',
#         norm_cfg=dict(type='BN', requires_grad=True)),
#     auxiliary_head=dict(
#         type='FCNHead',
#         norm_cfg=dict(type='BN', requires_grad=True)))

# custom_imports = dict(
#     imports=[
#         'mmcls.models.backbones.lympnet2'
#     ],
#     allow_failed_imports=False)

# Modify dataset related settings
dataset_type = 'LystoDataset'
data_root = PATH_DATASET

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_ = True,
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{PATH_DATASET}/train_IHC',
        ann_dir=f'{PATH_DATASET}/train_mask3',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size=(256, 256), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        _delete_ = True,
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{PATH_DATASET}/val_IHC',
        ann_dir=f'{PATH_DATASET}/val_mask3',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
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
        _delete_ = True,
        type=dataset_type,
        data_root=data_root,
        img_dir=f'{PATH_DATASET}/test_IHC',
        ann_dir=f'{PATH_DATASET}/test_mask3',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(224, 224),
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

data = dict(
    # samples_per_gpu=2,
    # workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix=f'{PATH_DATASET}/train_IHC',
        ann_file=f'{PATH_DATASET}/label_csvs/train_easy_vs_hard.csv'),
    val=dict(
        type=dataset_type,
        data_prefix=f'{PATH_DATASET}/val_IHC',
        ann_file=f'{PATH_DATASET}/label_csvs/val_easy_vs_hard.csv'),
    test=dict(
        type=dataset_type,
        data_prefix=f'{PATH_DATASET}/test_IHC',
        ann_file=f'{PATH_DATASET}/label_csvs/test_easy_vs_hard.csv'))

# custom_hooks = [
#     dict(
#         type='LymphCountEvalHook',
#         eval_interval=2,
#         file_prefix=MODEL_NAME,
#         path_config=PATH_CONFIG_FILE,
#         base_dir='evaluation1')
# ]

work_dir = PATH_WORK_DIR
# load_from = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
# load_from = os.path.join(PATH_WORK_DIR, 'latest.pth')
# resume_from = os.path.join(PATH_WORK_DIR, 'latest.pth')

total_epochs = max_epochs = 30
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001, nesterov=True)
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.0025, step=[12, 18])
# lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
evaluation = dict(interval=2, metric=['mIoU', 'mDice'], pre_eval=True)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=2)
dist_params = dict(backend='nccl')
cudnn_benchmark = True
seed = 0
gpu_ids = range(1)
workflow = [('train', 1), ('val', 1)]
