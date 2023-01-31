MMSEG_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation'
DATASET_HOME_PATH = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets'
S = 1
DATASET = 'lysto'
MODEL_NAME = 'fcn-unet-s5'
PATH_DATASET = f'./lymphocyte_dataset/{DATASET.upper()}-dataset/'
CONFIG_FILE_NAME = MODEL_NAME.replace('-', '_') + '_s' + str(S) + '_' + DATASET
PATH_CONFIG_FILE = f"./configs/fcn_lysto/{CONFIG_FILE_NAME}.py"
PATH_WORK_DIR = f'./trained_models/{DATASET}-models/{MODEL_NAME}/setting{S}/'

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

dataset_type = 'LystoDataset'
data_root = PATH_DATASET

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        _delete_ = True,
        type=dataset_type,
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='train_DAB_images',
        ann_dir='train_mask_images3',
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
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='val_DAB_images',
        ann_dir='val_mask_images3',
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
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='test_DAB_images1',
        ann_dir='test_mask_images3',
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

work_dir = PATH_WORK_DIR
# load_from = 'checkpoints/fcn_d6_r50-d16_512x1024_80k_cityscapes_20210306_115604-133c292f.pth'
# resume_from = PATH_WORK_DIR + '/epoch_10.pth'

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
runner = dict(type='IterBasedRunner', max_iters=50000)
evaluation = dict(interval=5000, metric=['mIoU', 'mDice'], pre_eval=True)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
checkpoint_config = dict(by_epoch=False, interval=5000)
dist_params = dict(backend='nccl')
cudnn_benchmark = True
seed = 0
gpu_ids = range(0, 1)
workflow = [('train', 1)]
