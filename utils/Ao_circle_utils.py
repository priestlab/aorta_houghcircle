from __future__ import division

import cv2
import numpy as np
from skimage.draw import circle as sk_draw_circle
from skimage.color import gray2rgb
from skimage.exposure import adjust_gamma

from utils.utils import get_square_border
from utils.utils import get_dice

def circle_distance(circle_i, circle_j):
    """
    Function for calculating the distance
    between the two circles.
    """
    cy_i, cx_i, rad_i, accum_i = circle_i
    cy_j, cx_j, rad_j, accum_j = circle_j
    distance = ((cy_i-cy_j)**2 + (cx_i-cx_j)**2)**.5
    return distance
    
def minDist_Pass(circle_, circle_collections, minDist):
    """
    Function for checking if the circle_ is in close
    proximity of the circle_collections
    """
    cy, cx, rad, accum = circle_
    for circle_i in circle_collections:
        cy_i, cx_i, rad_i, accum_i = circle_i
        if circle_distance(circle_, circle_i) < rad + rad_i + minDist:
            return False
    
    return True

def get_circle_image_patch(circle_, image_,
                           radDif=None, radVal=None):
    """
    Function for getting the circled image patch
    """
    try:
        cy, cx, rad, accum = circle_
        circle_mask_ = np.zeros_like(image_).astype(bool)
        
        if radVal is not None:
            rad_ = radVal
        elif radDif is not None:
            rad_ = max(rad + radDif, 4)
        else:
            rad_ = rad
            
        rr, cc = sk_draw_circle(cy, cx, rad_)
        circle_mask_[rr, cc] = True
        circle_img_ = image_[np.where(circle_mask_)]
        
        return circle_img_
    except:
        return None
    
def overlap_Pass(circle_, circle_collections):
    """
    Function for checking if the circle_ is overlapping
    with circles in circle_collections
    """
    return minDist_Pass(circle_, circle_collections, 0)

def circle_brightness_score(circle_, image_):
    """
    Function for getting the pixel intensity mean
    of the circled image patch.
    """
    circle_img_ = get_circle_image_patch(circle_, image_)

    if circle_img_ is not None:
        brightness_score = circle_img_.mean()
        return brightness_score
    else:
        return 0.0

def circle_brightness_Pass(circle_, image_, brightness_thresh):
    """
    Function for check if circled image patch
    pass the brightness_thresh
    """
    brightness_score_ = circle_brightness_score(circle_, image_)
    return brightness_score_ > brightness_thresh

def circle_location_Pass(circle_, image_, margin=0.15):
    """
    Function for check if the circle_ is overlapping
    with the margin of the image_.
    """
    cy, cx, rad, accum = circle_
    image_sizeY_, image_sizeX_ = image_.shape[0], image_.shape[1]
    margin_min_x = int(image_sizeX_ * margin)
    margin_max_x = int(image_sizeX_ * (1 - margin))
    margin_min_y = int(image_sizeY_ * margin)
    margin_max_y = int(image_sizeY_ * (1 - margin))
    
    margin_min_xh = int(image_sizeX_ * margin/2.)
    margin_max_xh = int(image_sizeX_ * (1 - margin/2.))
    margin_min_yh = int(image_sizeY_ * margin/2.)
    margin_max_yh = int(image_sizeY_ * (1 - margin/2.))
    
    if cy<margin_min_y or cy>margin_max_y:
        return False
    if cx<margin_min_x or cx>margin_max_x:
        return False
    
    if cy-rad<margin_min_yh or cy+rad>margin_max_yh:
        return False
    if cx-rad<margin_min_xh or cx+rad>margin_max_xh:
        return False
    
    return True
        
def circle_edge_noise_score(circle_, edge_, 
                            radDif=None, radVal=None):
    """
    Function for getting the sum of the circled patch
    of the edge image.
    """
    circle_img_ = get_circle_image_patch(circle_, edge_,
                                         radDif=radDif,
                                         radVal=radVal)
    
    if circle_img_ is not None:
        edge_noise_score = circle_img_.sum()
        return edge_noise_score
    else:
        return 1000.0

def circle_edge_noise_Pass(circle_, edge_, 
                           radDif=None, radVal=None, 
                           noise_thresh_=15):
    """
    Function for checking if the circled patch edge
    image is over the edge noise threshold
    """
    noise_score_ = circle_edge_noise_score(circle_, edge_, radDif, radVal)
    return noise_score_ < noise_thresh_

def circle_image_std_score(circle_, image_,
                           radDif=None, radVal=None):
    """
    Function for getting the circled patch image
    standard deviation
    """
    circle_img_ = get_circle_image_patch(circle_, image_,
                                         radDif=radDif,
                                         radVal=radVal)
    
    if circle_img_ is not None:
        image_std_score = circle_img_.std()
        return image_std_score
    else:
        return 1000.0

def circle_image_std_Pass(circle_, image_, 
                          radDif=None, radVal=None,
                          std_thresh_=50):
    """
    Function for checking if the circled image patch
    is over the intensity standard deviation threshold
    """
    std_score_ = circle_image_std_score(circle_, image_,
                                        radDif=radDif,
                                        radVal=radVal)
    return std_score_ < std_thresh_


def watershed_dice_score(circle_, image, gamma=2.5):
    try:
        cy, cx, rad, accum = circle_
        
        marker = np.zeros_like(image)
        rr, cc = sk_draw_circle(cy, cx, max(rad-2, 2))
        border = get_square_border(image, cut=0.07)
        
        marker[rr,cc] = 50
        marker[border == True] = 255
        
        circle_mask = np.zeros_like(image)
        rr, cc = sk_draw_circle(cy, cx, rad)
        circle_mask[rr,cc] = 1
        
        image = adjust_gamma(image, gamma=gamma)
        if image.ndim == 2:
            image = gray2rgb(image)
        
        segmentation_ = cv2.watershed(image, marker.astype(np.int32))
        segmentation_ = (segmentation_ == 50).astype(image.dtype)
        
        watershed_score = get_dice(segmentation_, circle_mask)
        
        #return watershed_score, segmentation_, circle_mask
        return watershed_score
    except:
        return 0.0

def watershed_dice_score_Pass(circle_, image, dice_thresh=0.7, gamma=2.5):
    dice_score = watershed_dice_score(circle_, image, gamma=gamma)
    return dice_score > dice_thresh
