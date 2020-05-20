from __future__ import division

import math
import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import circle_perimeter
from skimage.color import gray2rgb

from utils.utils import get_border

def get_overlay_circles(image, circles):
    image_rgb = gray2rgb(image)
    for circle_i in circles:
        cy, cx, radius, accum = circle_i
        print("center: [{},{}], radius: {}".format(cy, cx, radius))
        rr, cc = circle_perimeter(cy, cx, radius)
        image_rgb[rr, cc] = (220, 20, 20)

    return image_rgb

def get_watershed_overlay(image, mask):
    if image.ndim == 2:
        image = gray2rgb(image)
    image[mask==-1] = (255,0,0)

    return image

def get_marker_overlay(image, marker):
    image = gray2rgb(image)
    image[(marker>0)*(marker<255)] = (255, 0, 0)

    return image

def floodFill_to_watershed(mask, mVal):
    seg = mask == mVal
    seg_border = get_border(seg, iterations=1)

    newMask = np.zeros_like(mask).astype(np.int32)
    newMask[seg] = mVal
    newMask[seg_border] = -1
    newMask[np.where(newMask==0)] = 255

    return newMask



def get_floodFill_overlay(image, mask):
    seg = mask > 1
    seg_border = get_border(seg, iterations=1)

    overlay = np.copy(image)
    if overlay.ndim == 2:
        overlay = gray2rgb(overlay)
    overlay[seg_border] = (255,0,0)
    
    return overlay

def display_image_titles(images_w_titles, n_cols=3):
    """
    Display the images with titles
    """
    n_img = len(images_w_titles)
    n_rows = math.ceil(n_img * 1. / n_cols)
    plt.figure(figsize=(16, (16/n_cols)*n_rows))
    for i, (image, title) in enumerate(images_w_titles):
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.imshow(image)
        ax.set_title(title)

    plt.show()

