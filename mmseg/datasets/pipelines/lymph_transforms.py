import cv2
import numpy as np

from skimage.color import rgb2hed, hed2rgb

from ..builder import PIPELINES

def normalize_255(image):
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    return image.astype(np.uint8)

@PIPELINES.register_module()
class IHC2DAB(object):
    """ Convert IHC Image to DAB Image """

    def __call__(self, results):
        print(results)
        img_ihc_bgr = results['img']
        img_ihc_rgb = cv2.cvtColor(img_ihc_bgr, cv2.COLOR_BGR2RGB)
        img_hed_rgb = rgb2hed(img_ihc_rgb)
        null = np.zeros_like(img_hed_rgb[:, :, 0])
        # ihc_h = hed2rgb(np.stack((img_hed_rgb[:, :, 0], null, null), axis=-1))
        # ihc_e = hed2rgb(np.stack((null, img_hed_rgb[:, :, 1], null), axis=-1))
        img_dab_rgb = hed2rgb(np.stack((null, null, img_hed_rgb[:, :, 2]), axis=-1))
        img_dab_rgb = normalize_255(img_dab_rgb)
        img_dab_bgr = cv2.cvtColor(img_dab_rgb, cv2.COLOR_BGR2RGB)
        results['img'] = img_dab_bgr
        return results

    def __repr__(self):
        return (self.__class__.__name__ + '()')

@PIPELINES.register_module()
class IHC2HSV(object):
    """ Convert IHC Image to HSV Image """

    def __call__(self, results):
        img = results['img']
        results['img'] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return results

    def __repr__(self):
        return (self.__class__.__name__ + '()')
