import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mmdet.apis import inference_detector, init_detector
from mmseg.apis import inference_segmentor, init_segmentor
from mmdet.utils.lysto_utils import ihc2dab, ihc2hsv

def get_lysto_image_path():
    base_path = '/home/gpu02/maskrcnn-lymphocyte-detection/datasets/lysto'
    path_images = os.path.join(base_path, 'train_DAB')
    path_masks = os.path.join(base_path, 'train_mask1')
    image_names = os.listdir(path_images)
    path_image_rgb = os.path.join(path_images, image_names[10])
    print('[path_image_rgb]', path_image_rgb)
    image_rgb = cv2.imread(path_image_rgb)
    path_image_mask = os.path.join(path_masks, image_names[10])
    print('[path_image_mask]', path_image_mask)
    image_mask = cv2.imread(path_image_mask)
    print('[image_mask]', image_mask.shape, np.unique(image_mask))
    image_mask[image_mask == 1] = 255
    return image_rgb, image_mask, path_image_rgb

def get_lyon_image_path(index=-1):
    base_path = 'lymphocyte_dataset/LYON-dataset'
    path_images = os.path.join(base_path, 'Train')
    path_masks = os.path.join(base_path, 'Train_binary')
    image_names = os.listdir(path_images)
    selected_name = np.random.choice(image_names) if index < 0 or index >= len(image_names) else image_names[index]
    path_image_rgb = os.path.join(path_images, selected_name)
    print('[path_image_rgb]', path_image_rgb)
    image_rgb = cv2.imread(path_image_rgb)
    path_image_mask = os.path.join(path_masks, selected_name)
    print('[path_image_mask]', path_image_mask)
    image_mask = cv2.imread(path_image_mask)
    print('[image_mask]', image_mask.shape, np.unique(image_mask))
    image_mask[image_mask != 0] = 255
    print('[image_mask]', image_mask.shape, np.unique(image_mask))
    return image_rgb, image_mask, path_image_rgb

# config_file = './configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py'
# checkpoint_file = './checkpoints/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth'
config_file = '/home/gpu02/maskrcnn-lymphocyte-detection/mmsegmentation/configs/lysto/seglymphnet3_s1.py'
# checkpoint_file = './trained_models/lyon-models/maskrcnn-lymphocytenet5-cm3/setting13/epoch_25.pth'
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file)
# print(model)

# test a single image
image_rgb, image_mask, path_image_rgb = get_lysto_image_path()

# image_dab = ihc2dab(image_rgb)
image_hsv = ihc2hsv(image_rgb)

result = inference_segmentor(model, path_image_rgb)
detections, masks = result[0][0], np.array(result[1][0], dtype='uint8')
print('[detections]', detections.shape)
print('[masks]', masks.shape, np.unique(masks))

mask = masks.sum(axis=0).astype('uint8')
mask[mask != 0] = 255
mask = np.stack((mask, mask, mask), axis=2)
print('[mask]', mask.shape, np.unique(mask))

fig = plt.figure(figsize=(10, 6))

ax1 = plt.subplot(2, 3, 1)
ax1.title.set_text('Original RGB')
plt.imshow(image_rgb)

ax2 = plt.subplot(2, 3, 2)
ax2.title.set_text('Original HSV')
plt.imshow(image_hsv)

# ax3 = plt.subplot(2, 3, 3)
# ax3.title.set_text('Original DAB')
# plt.imshow(image_dab)

ax4 = plt.subplot(2, 3, 4)
ax4.title.set_text('Original Mask')
plt.imshow(image_mask, cmap='gray')

ax5 = plt.subplot(2, 3, 5)
ax5.title.set_text('Predicted Mask')
plt.imshow(mask, cmap='gray')

plt.show()
