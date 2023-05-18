import cv2
import numpy as np

from skimage.color import rgb2hed, hed2rgb

from ..builder import PIPELINES

def normalize_255(image):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    return image.astype(np.uint8)


def ihc2dab(img_ihc_bgr):
    img_ihc_rgb = cv2.cvtColor(img_ihc_bgr, cv2.COLOR_BGR2RGB)
    img_hed_rgb = rgb2hed(img_ihc_rgb)
    null = np.zeros_like(img_hed_rgb[:, :, 0])
    img_dab_rgb = hed2rgb(np.stack((null, null, img_hed_rgb[:, :, 2]), axis=-1))
    img_dab_rgb = normalize_255(img_dab_rgb)
    img_dab_bgr = cv2.cvtColor(img_dab_rgb, cv2.COLOR_BGR2RGB)
    return img_dab_bgr


@PIPELINES.register_module()
class IHC2DAB(object):
    """ Convert IHC Image to DAB Image """

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, results):
        apply_transform = True if np.random.rand() <= self.p else False
        results['IHC2DAB'] = False
        if apply_transform:
            img_ihc_bgr = results['img']
            img_dab_bgr = ihc2dab(img_ihc_bgr)
            results['img'] = img_dab_bgr
            results['IHC2DAB'] = True
        return results

    def __repr__(self):
        return (self.__class__.__name__ + f'(p={self.p})')


@PIPELINES.register_module()
class IHC2HSV(object):
    """ Convert IHC Image to HSV Image, also change the H channel
        from [0, 179] range to [0, 255] range """

    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, results):
        apply_transform = True if np.random.rand() <= self.p else False
        results['IHC2HSV'] = False
        if apply_transform:
            img = results['img']
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = hsv[:, :, 0] * 255 / 179
            results['img'] = hsv
            results['IHC2HSV'] = True
        return results

    def __repr__(self):
        return (self.__class__.__name__ + f'(p={self.p})')
