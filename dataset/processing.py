import cv2
import numbers
import numpy as np
from torchvision.transforms import Compose
from skimage.util import pad as skimage_pad
from skimage.exposure import rescale_intensity

import logging

logger = logging.getLogger(__name__)

class StdNormalize(object):
    def __call__(self, image, mask):
        if image.ndim == 3:
            if image.shape[-1] == 3 or image.shape[-1] == 1:
                image = image.transpose(2, 0, 1)
            image_norm = np.array([(x-x.mean())/x.std() for x in image])
        else:
            image_norm = (image - image.mean())/image.std()

        return image_norm.astype(np.float32), mask.astype(np.float32)

class SizeCrop(object):
    # Crop the image on each side, no stretching
    def __init__(self, crop=(0.1, 0.1, 0.1, 0.1)):
        self.crop_h = crop[:2]
        self.crop_w = crop[2:]

    def get_size(self, img):
        h, w = img.shape
        h0, h1 = [int(x * h) for x in self.crop_h]
        w0, w1 = [int(x * w) for x in self.crop_w]

        return (h0, h1, w0, w1)

    def __call__(self, image, mask):
        h0, h1, w0, w1 = self.get_size(image)

        cropped_image = image[h0:-h1, w0:-w1]
        cropped_mask = mask[h0:-h1, w0:-w1]

        return cropped_image, cropped_mask

    def __repr__(self):
        return self.__class__.__name__ + '(crop={})'.format(self.crop_h + self.crop_w)

class PadCrop(object):
    # Pad image if necessary, then crop, no stretching
    def __init__(self, size=(200, 200), mode="constant"):
        self.size   = size
        self.mode   = mode

    def pad(self, img, mask):
        h, w    = img.shape
        pad_h   = (self.size[0] - h) if (h < self.size[0]) else 0
        pad_w   = (self.size[1] - w) if (w < self.size[1]) else 0

        if pad_h + pad_w == 0:
            return img, mask

        pad_h0  = int(pad_h / 2)
        pad_h1  = pad_h - pad_h0
        pad_w0  = int(pad_w / 2)
        pad_w1  = pad_w - pad_w0
        img     = skimage_pad(img, ((pad_h0, pad_h1),(pad_w0, pad_w1)), self.mode)
        mask    = skimage_pad(mask, ((pad_h0, pad_h1),(pad_w0, pad_w1)), self.mode)

        return img, mask

    def get_offsets(self, img):
        h, w = img.shape
        offset_h = int((h - self.size[0]) / 2)
        offset_w = int((w - self.size[1]) / 2)

        return (offset_h, offset_w)

    def __call__(self, image, mask):
        image, mask = self.pad(image, mask)
        offset1, offset2 = self.get_offsets(image)

        cropped_image = image[offset1:offset1+self.size[0],
                              offset2:offset2+self.size[1]]
        cropped_mask = mask[offset1:offset1+self.size[0],
                            offset2:offset2+self.size[1]]

        return cropped_image, cropped_mask

    def __repr__(self):
        return self.__class__.__name__ + '(size={} | model={})'.format(self.size, self.mode)


class Resize(object):
    def __init__(self, size=(128, 128)):
        if isinstance(size, numbers.Number):
            self.size = tuple([int(size), int(size)])
        elif isinstance(size, tuple) or isinstance(size, list):
            self.size = tuple([int(x) for x in size])
        else:
            raise Exception("Resize() only takes input args such as number, tuple, list.\n"\
                            "Wrong Input: {}".format(size))

    def __call__(self, image, mask):
        resized_image = cv2.resize(image, self.size, interpolation = cv2.INTER_CUBIC)
        resized_mask = cv2.resize(mask.astype(float), self.size, interpolation = cv2.INTER_NEAREST)
        return resized_image, resized_mask

    def resize_image(self, image):
        resized_image = cv2.resize(image, self.size, interpolation = cv2.INTER_CUBIC)
        return resized_image

    def resize_mask(self, mask):
        resized_mask = cv2.resize(mask.astype(float), self.size, interpolation = cv2.INTER_NEAREST)
        return resized_mask
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class CropResize(object):
    # Crop the image first to square shape, then resize
    def __init__(self, size=(128, 128)):
        self.name = 'CropResize'
        self.size = size
        self.resize = Resize(size)

    def crop(self, img):
        h, w = img.shape
        crop_h0, crop_h1 = 0, h
        crop_w0, crop_w1 = 0, w
        if h > w:
            crop_h0 = int((h - w)/2)
            crop_h1 = crop_h0 + w
        elif w > h:
            crop_w0 = int((w - h)/2)
            crop_w1 = crop_w0 + h
        
        return img[crop_h0:crop_h1, crop_w0:crop_w1]

    def __call__(self, image, mask):
        image = self.crop(image)
        mask = self.crop(mask)
        image, mask = self.resize(image, mask)

        return image, mask
    def crop_resize_image(self, image):
        image = self.crop(image)
        image = self.resize.resize_image(image)

        return image

    def crop_resize_mask(self, mask):
        mask = self.crop(mask)
        mask = self.resize.resize_mask(mask)

        return mask

def ResizePad(image, mask):
    if image.shape == mask.shape:
        return mask

    h, w = image.shape
    side = min(h, w)
    mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)

    pad_h0, pad_h1 = 0, 0
    pad_w0, pad_w1 = 0, 0

    if h > w:
        pad_h0 = int((h-w)/2)
        pad_h1 = h - w - pad_h0
    elif w > h:
        pad_w0 = int((w-h)/2)
        pad_w1 = w - h - pad_w0

    padding = ((pad_h0, pad_h1),(pad_w0, pad_w1))
    mask = skimage_pad(mask, padding, 'constant')
    mask = rescale_intensity(mask.astype(np.float64), out_range=(0.0, 1.0))

    return mask

#-----------------------------------------------------------------------------
# Preprocess steps
#-----------------------------------------------------------------------------
PREPROCESS = dict(SizeCrop=SizeCrop,
                  CropResize=CropResize)

class Preprocess(object):
    def __init__(self, **kwargs):
        self.preprocesses   = []

        logger.info("===========================")
        logger.info("Preprocess configs")
        logger.info("===========================")
        for prep in kwargs.keys():
            try:
                preprocess = PREPROCESS[prep](**kwargs[prep])
                self.preprocesses.append(preprocess)
                logger.info(" :{}".format(prep))
                for key in kwargs[prep].keys():
                    logger.info("{:>25}: {}".format(key, kwargs[prep][key]))
            except Exception as err:
                #logger.info("PREPROCESS | Error: {}".format(err))
                pass

        logger.info("# of steps in preprocessing: {}\n".format(len(self.preprocesses)))
    
    def __call__(self, image, mask):
        for p in self.preprocesses:
            image, mask = p(image, mask)
        return image, mask

#-----------------------------------------------------------------------------
# Postprocess steps
#-----------------------------------------------------------------------------
POSTPROCESS = dict(Resize=Resize,
                   CropResize=CropResize,
                   PadCrop=PadCrop,
                   StdNormalize=StdNormalize)

class Postprocess(object):
    def __init__(self, **kwargs):
        self.postprocesses   = []

        logger.info("===========================")
        logger.info("Postprocess configs")
        logger.info("===========================")
        for post in sorted(kwargs.keys()):
            try:
                postprocess = POSTPROCESS[post](**kwargs[post])
                self.postprocesses.append(postprocess)
                logger.info(" :{}".format(post))
                for key in kwargs[post].keys():
                    logger.info("{:>25}: {}".format(key, kwargs[post][key]))
            except Exception as err:
                #logger.info("POSTPROCESS | Error: {}".format(err))
                pass

        logger.info("# of steps in postprocessing: {}\n".format(len(self.postprocesses)))

    def __call__(self, image, mask):
        for p in self.postprocesses:
            image, mask = p(image, mask)
        return image, mask


