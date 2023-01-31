import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor

# config_file = './configs/unet/fcn_unet_s5-d16_256x256_40k_hrf.py'
# checkpoint_file = './checkpoints/fcn_unet_s5-d16_256x256_40k_hrf_20201223_173724-d89cf1ed.pth'
config_file = './configs/lysto/fcn_unet_s5_s1_lysto.py'
checkpoint_file = './trained_models/lysto-models/fcn-unet-s5/setting1/iter_50000.pth'
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file)
# print(model)

# test a single image
# img = './demo/demo.png'
# img = './data/HRF/images/validation/06_dr.png'
base_path = 'lymphocyte_dataset/LYSTO-dataset'
path_images = os.path.join(base_path, 'train_DAB_images')
path_masks = os.path.join(base_path, 'train_mask_images3')
image_names = os.listdir(path_images)
path_image_rgb = os.path.join(path_images, image_names[10])
print('[path_image_rgb]', path_image_rgb)
image_rgb = cv2.imread(path_image_rgb)
path_image_mask = os.path.join(path_masks, image_names[10])
print('[path_image_mask]', path_image_mask)
image_mask = cv2.imread(path_image_mask)
print('[image_mask]', image_mask.shape, np.unique(image_mask))
image_mask[image_mask == 1] = 255

result = inference_segmentor(model, path_image_rgb)
img_result = result[0] #.astype(np.uint8)
print('[img_result]', img_result.shape, img_result.dtype)

fig = plt.figure(figsize=(10, 6))

ax1 = plt.subplot(1, 3, 1)
ax1.title.set_text('Original')
plt.imshow(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB))

ax2 = plt.subplot(1, 3, 2)
ax2.title.set_text('GT Mask')
plt.imshow(image_mask, cmap='gray')

ax3 = plt.subplot(1, 3, 3)
ax3.title.set_text('Prediction')
plt.imshow(img_result, cmap='gray')

plt.show()
