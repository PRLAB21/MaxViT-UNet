import numpy as np
import torch
from mmseg.models.backbones import EfficientUNet, UNet, VisionTransformer, SegLymphNet3, SegLymphNet, SegLymphNet2
from mmseg.models.segmentors import EncoderDecoder

def calculate_weights(model):
    weights = []
    for name, param in model.named_parameters():
        # print(name, param.size(), np.prod(param.size()))
        weights.append(np.prod(param.size()))
    return np.sum(weights)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

inputs = torch.rand(2, 3, 256, 256).to(device)
# model = LymphocyteNet3_CM1(fusion_type='concat', debug=True)
# model = EfficientUNet(model_name='efficientnet-b0', debug=True)
# model = UNet()
# model = VisionTransformer(out_indices=(2, 5, 8, 11))

# model = EncoderDecoder(
#     backbone=dict(
#         type='VisionTransformer',
#         img_size=(256, 256),
#         out_indices=(2, 5, 8, 11),
#         norm_cfg=dict(type='LN', eps=1e-06)),
#     neck=dict(
#         type='MultiLevelNeck',
#         in_channels=[768, 768, 768, 768],
#         out_channels=768,
#         scales=[4, 2, 1, 0.5]),
#     decode_head=dict(
#         type='UPerHead',
#         in_channels=[768, 768, 768, 768],
#         in_index=[0, 1, 2, 3],
#         pool_scales=(1, 2, 3, 6),
#         channels=512,
#         dropout_ratio=0.1,
#         num_classes=150,
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
#     auxiliary_head=dict(
#         type='FCNHead',
#         in_channels=768,
#         in_index=3,
#         channels=256,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         num_classes=150,
#         norm_cfg=dict(type='SyncBN', requires_grad=True),
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#     train_cfg=dict(),
#     test_cfg=dict(mode='whole'))

model = EncoderDecoder(
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
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# model = SegLymphNet(debug=True)
# model = SegLymphNet2(debug=True)
# exit()
print('-' * 25)
print(model)
print('-' * 25)

model = model.to(device)
model.eval()
backbone_outputs = model.backbone(inputs)
for i, level in enumerate(backbone_outputs):
    print(f'level{i} -> {backbone_outputs[i].shape}')

neck_outputs = model.neck(backbone_outputs)
for i, level in enumerate(neck_outputs):
    print(f'level{i} -> {neck_outputs[i].shape}')

decode_head_outputs = model.decode_head(neck_outputs)
print('[decode_head_outputs]', decode_head_outputs.shape)

auxiliary_head_outputs = model.auxiliary_head(neck_outputs)
print('[auxiliary_head_outputs]', auxiliary_head_outputs.shape)

backbone_outputs = model.backbone(inputs)

print('model.align_corners:', model.align_corners)
print('model.num_classes:', model.num_classes)
print('model.out_channels:', model.out_channels)

print(calculate_weights(model) / 1e6)
