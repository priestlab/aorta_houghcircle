from __future__ import division

import cv2
import numpy as np
from skimage.draw import circle as sk_draw_circle
from skimage.color import gray2rgb
from skimage.exposure import adjust_gamma
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, binary_closing

from utils.utils import get_square_border
from utils.utils import get_edge_mask
from utils.utils import get_center_blob_border
from utils.display import get_marker_overlay
from utils.display import floodFill_to_watershed

############################################################
# Watershed Circles
############################################################
def watershed_circle(image, circle, mVal=50, gamma=2.5):
    cy, cx, rad, accum = circle
    
    marker = np.zeros_like(image)
    rr, cc = sk_draw_circle(cy, cx, max(rad-2, 2))
    border = get_square_border(image, cut=0.07)
    
    marker[rr,cc] = mVal
    marker[border == True] = 255
    
    image = adjust_gamma(image, gamma=gamma)
    image = gray2rgb(image)
    segmentation_mask = cv2.watershed(image, marker.astype(np.int32))
    
    return segmentation_mask, marker

def combine_watershed_mask(MaskC, mask, mVal):
    MaskC[np.where(mask==-1)] = -1
    MaskC[np.where(mask==mVal)] = mVal
    
    return MaskC

def watershed_all_circles(image, circles, labelValue, gamma=2.5):
    circle_seg = np.zeros_like(image).astype(np.int32)
    circle_marker = np.zeros_like(image)
    for i, circle_i in enumerate(circles):
        circle_seg_i, circle_marker_i = watershed_circle(image, circle_i, mVal=labelValue[i], gamma=gamma)
        circle_seg = combine_watershed_mask(circle_seg, circle_seg_i, labelValue[i])
        circle_marker = combine_watershed_mask(circle_marker, circle_marker_i, labelValue[i])

    circle_seg[np.where(circle_seg==0)] = 255
    circle_marker_overlay = get_marker_overlay(image, circle_marker)
    return circle_seg, circle_marker_overlay

def get_all_circle_masks(image, circles, labelValue):
    #circle_seg = np.zeros_like(image).astype(np.int32)
    #for i, circle_i in enumerate(circles):
    #    circle_seg_i = 
    #    circle_seg = combine_watershed_mask(circle_seg, circle_seg_i, labelValue[i])

    #circle_seg[np.where(circle_seg==0)] = 255
    #circle_marker_overlay = get_marker_overlay(image, circle_marker)
    #return circle_seg, circle_marker_overlay
    return None, None


def segment_all_circles(image, circles, labelValue, wFlag=True):
    if wFlag:
        return watershed_all_circles(image, circles, labelValue)
    else:
        return get_all_circle_masks(image, circles, labelValue)


############################################################
# Watershed Pulmonary
############################################################
def get_pulmonary_seedMarker(image, seedPoint, mVal):
    seedMarker = np.zeros_like(image)
    border = get_square_border(image, cut=0.07)
    seedMarker[border==True] = 255
    seedMarker[seedPoint[0], seedPoint[1]:seedPoint[1]+5] = mVal 

    return seedMarker

def watershed_pulmonary_marker(image, seedMarker, gamma=2.5):
    image = adjust_gamma(image, gamma=gamma)
    image = gray2rgb(image)
    p_seg = cv2.watershed(image, seedMarker.astype(np.int32))

    p_marker = np.copy(image).astype(np.uint8)
    p_marker[(seedMarker>0)*(seedMarker<255)] = (255, 0, 0)
    
    return p_seg, p_marker



############################################################
# FloodFill Pulmonary
############################################################
def floodFillPulmonary(image, seedPoint, mVal=150, use_center_border=False):
    seedPoint = (seedPoint[1], seedPoint[0])
    
    # get the floodFill edge image
    img_sobel = sobel(image)
    thresh = threshold_otsu(img_sobel)
    edge_mask = get_edge_mask(image, 9)
    
    edge_image= img_sobel * edge_mask > thresh
    #TEST
    if use_center_border:
        center_border = get_center_blob_border(image)
        edge_image[center_border] = True
    #TEST
    edge_image = binary_closing(edge_image).astype(np.uint8)

    flood_mask = np.zeros_like(edge_image)
    
    # get the edge image with marker
    edge_image_rgb = gray2rgb(edge_image*255)
    edge_image_rgb[seedPoint[1],seedPoint[0]-10:seedPoint[0]+10] = (255, 0, 0)
    edge_image_rgb[seedPoint[1]-10:seedPoint[1]+10, seedPoint[0]] = (255, 0, 0)
    
    # floodFill
    cv2.floodFill(edge_image[1:-1,1:-1], flood_mask, seedPoint=seedPoint, newVal=mVal, flags=(4 | ( mVal << 8 )))
    flood_mask = floodFill_to_watershed(flood_mask, mVal)

    return flood_mask, edge_image_rgb

