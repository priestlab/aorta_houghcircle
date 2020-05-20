from __future__ import division

import cv2
import numpy as np
from math import ceil
from skimage.color import gray2rgb
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu



##################################################
## Tools
## get_border (mask)
## get_center_marker (mask, marker_iter=3)
## get_square_border (image, cut=0.2, iterations=1)
## get_edge_mask (image, edge=26)
##################################################

def get_border(mask, iterations=5):
    border = cv2.dilate(mask.astype(np.uint8), None, iterations=iterations) - \
             cv2.dilate(mask.astype(np.uint8), None, iterations=iterations-1)
    return border.astype(bool)

def get_center_marker(mask, marker_iter=3):
    c_pt = center_of_mass(mask)
    c_pt = tuple([int(x) for x in c_pt])
    c_marker = np.zeros_like(mask).astype(np.uint8)
    c_marker[c_pt] = 1
    c_marker = cv2.dilate(c_marker, None, iterations=marker_iter).astype(bool)
    return c_marker

def get_square_border(image, cut=0.1, iterations=1):
    border = np.zeros_like(image)
    px = ceil(cut * image.shape[0])
    py = ceil(cut * image.shape[1])
    border[px:-px, py] = 1
    border[px:-px, -py] = 1
    border[px, py:-py] = 1
    border[-px, py:-py] = 1
    border = cv2.dilate(border, None, iterations=iterations).astype(bool)
    return border

def get_edge_mask(image, edge=26):
    edge_mask = np.zeros_like(image).astype(bool)
    edge_mask[edge:-edge, edge:-edge] = True
    return edge_mask


def get_square_border(image, cut=0.2, iterations=1):
    border = np.zeros_like(image)
    px = int(ceil(cut * image.shape[0]))
    py = int(ceil(cut * image.shape[1]))
    border[px:-px, py] = 1
    border[px:-px, -py] = 1
    border[px, py:-py] = 1
    border[-px, py:-py] = 1
    border = cv2.dilate(border, None, iterations=iterations).astype(bool)
    return border

def get_dice(y_true, y_pred, epsilon=1e-15, labels=[0, 1]):
    y_true = y_true.round()
    y_pred = y_pred.round()
    intersection = (y_true * y_pred).sum(-1).sum(-1).sum(-1)
    union = y_true.sum(-1).sum(-1).sum(-1) + y_pred.sum(-1).sum(-1).sum(-1)
    
    return (2 * intersection + epsilon) / (union + epsilon)

def organize_circles(circles):
    if len(circles) == 2:
        if (circles[0][0]**2.+circles[0][1]**2.)**.5 > (circles[1][0]**2.+circles[1][1]**2.)**.5:
            circles = [circles[1], circles[0]]

    return circles

def fill_pul_seg_holes(p_seg, mVal):
    p_seg_b = p_seg==mVal
    p_seg_b_filled = ndi.binary_fill_holes(p_seg_b)
    p_seg_b_filled_border = get_border(p_seg_b_filled, iterations=1)

    newMask = np.zeros_like(p_seg).astype(np.int32)
    newMask[p_seg_b_filled] = mVal
    newMask[p_seg_b_filled_border] = -1
    newMask[np.where(newMask==0)] = 255

    return newMask
    
def filter_pseg_w_cseg(p_seg, c_seg):

    c_seg_b = (c_seg>1)*(c_seg<255)
    p_seg_b = (p_seg>1)*(p_seg<255)
    mVal = np.unique(p_seg[p_seg_b])[0]

    c_seg_b_dilated = cv2.dilate(c_seg_b.astype(np.uint8), None, iterations=1).astype(bool)
    p_seg_b[c_seg_b_dilated] = False
    p_seg_b_border = get_border(p_seg_b, iterations=1)

    newMask = np.zeros_like(p_seg).astype(np.int32)
    newMask[p_seg_b] = mVal
    newMask[p_seg_b_border] = -1
    newMask[np.where(newMask==0)] = 255

    return newMask


def get_center_blob_border(image, area_thresh=400):
    otsu_thresh = threshold_otsu(image)
    binary_image = image > otsu_thresh
    binary_image = ndi.binary_fill_holes(binary_image)
    label_image, n_labels = ndi.label(binary_image)
    label_areas = np.array([(label_image==x).sum() for x in range(n_labels)])
    label_passed, = np.where(label_areas>area_thresh)
    label_areas_index = label_passed[np.argsort(label_areas[label_passed])][::-1]
    for index_i in label_areas_index:
        blob_img = label_image == index_i
        rr_, cc_ = np.where(blob_img)
        blob_max = max(rr_.max(), cc_.max())
        blob_min = min(rr_.min(), cc_.min())
        rr_dist = rr_.max() - rr_.min()
        cc_dist = cc_.max() - cc_.min()
        if blob_max == 127 or blob_min == 0:
            continue
        if (rr_dist > image.shape[0]*0.8 or cc_dist > image.shape[1]*0.8):
            continue
        blob_border = get_border(blob_img)
        return blob_border
    
    return None
