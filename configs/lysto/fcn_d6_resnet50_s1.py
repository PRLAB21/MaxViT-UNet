S = 1
DATASET = 'lysto'
MODEL_NAME = 'fcn-d6-resnet50'
PATH_DATASET = f'./lymphocyte_dataset/{DATASET.upper()}-dataset/'
CONFIG_FILE_NAME = MODEL_NAME.replace('-', '_') + '_s' + str(S) + '_' + DATASET
PATH_CONFIG_FILE = f"./configs/fcn_lysto/{CONFIG_FILE_NAME}.py"
PATH_WORK_DIR = f'./trained_models/{DATASET}-models/{MODEL_NAME}/setting{S}/'

_base_ = [
    '../_base_/models/fcn_r50-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

model = dict(
    backbone=dict(dilations=(1, 1, 1, 2), strides=(1, 2, 2, 1), init_cfg=None),
    decode_head=dict(dilation=6, num_classes=1),
    auxiliary_head=dict(dilation=6, num_classes=1))

dataset_type = 'LystoDataset'
# classes = ('lymphocyte',)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='train_DAB_images',
        ann_dir='train_circular_masks',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type=dataset_type,
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='test_DAB_images1',
        ann_dir='test_circular_masks',
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
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type=dataset_type,
        # classes=classes,
        data_root=PATH_DATASET,
        img_dir='test_DAB_images1',
        ann_dir='test_circular_masks',
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
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

work_dir = PATH_WORK_DIR
# load_from = 'checkpoints/fcn_d6_r50-d16_512x1024_80k_cityscapes_20210306_115604-133c292f.pth'
# resume_from = PATH_WORK_DIR + '/epoch_10.pth'

max_iters = 2500
runner = dict(type='IterBasedRunner', max_iters=max_iters)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
evaluation = dict(interval=10, metric=['mIoU', 'mDice'], pre_eval=True)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001)
checkpoint_config = dict(interval=500, by_epoch=False)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
seed = 0
gpu_ids = range(1)
workflow = [('train', 1), ('val', 1)]

# total_epochs = 1
# runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=total_epochs)
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# # evaluation = dict(metric=['bbox', 'segm'], interval=100)
# evaluation = dict(interval=1, metric='mIoU', pre_eval=True)
# lr_config = dict(_delete_=True, policy='poly', power=0.9, min_lr=0.0001)
# checkpoint_config = dict(_delete_=True, interval=2)
# log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
# seed = 0
# gpu_ids = range(1)
# workflow = [('train', 1), ('val', 1)]
