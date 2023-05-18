# model settings
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
        mlp_ratio=4.,
        channels=64,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=5.0, ignore_index=0),
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
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=5.0, ignore_index=0),
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(170, 170)))
