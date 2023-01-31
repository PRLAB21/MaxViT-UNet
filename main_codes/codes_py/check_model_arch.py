import numpy as np
import torch
from mmseg.models.backbones import EfficientUNet, UNet, SegLymphNet3, SegLymphNet, SegLymphNet2

def calculate_weights(model):
    weights = []
    for name, param in model.named_parameters():
        # print(name, param.size(), np.prod(param.size()))
        weights.append(np.prod(param.size()))
    return np.sum(weights)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

inputs = torch.rand(4, 3, 256, 256).to(device)
# model = LymphocyteNet3_CM1(fusion_type='concat', debug=True)
# model = EfficientUNet(model_name='efficientnet-b0', debug=True)
# model = UNet()
# model = SegLymphNet(debug=True)
model = SegLymphNet2(debug=True)
# exit()
print('-' * 25)
print(model)
print('-' * 25)

model = model.to(device)
model.eval()
level_outputs = model(inputs)
# print(level_outputs.shape)
for i, level in enumerate(level_outputs):
    print(f'level{i} -> {level_outputs[i].shape}')
print(calculate_weights(model) / 1e6)
