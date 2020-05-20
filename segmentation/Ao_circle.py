from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks

from utils.Ao_circle_utils import *



## filter with brightness Pass, location Pass, then sort by Edge and Noise Pass
## Then select the overlap of the topX of Edge Pass and Noise Pass
## Then apply the overlap Pass and MinDist Pass
def Ao_hough_circle0(image, radmin=8, radmax=15, radint=2, n_circles=2, minDist=10,
                     gaussian_sigma=1.2, radDif=None, radVal=None, margin=0.15,
                     accum_thresh=0.7, noise_thresh=20, std_thresh=40,
                     dice_thresh=0.7):
    """
    Params
    ------
    image: input_image for finding circles
    radmin: min_radius
    radmax: max_radius
    radint: radius_interval
    n_circles: number_of_circles to be located
    minDist: minimum distance between each pair of circles
    
    Return
    ------
    """
        
            
    # apply gaussian filter, then rescale the image intensity
    _image = rescale_intensity(gaussian(image, sigma=gaussian_sigma), out_range=np.uint8).astype(np.uint8)
    
    # Image Otsu Threshold Mask Mean
    image_otsu_thresh = threshold_otsu(_image)
    print("Thresh: {}".format(image_otsu_thresh))
    otsu_mask = _image>image_otsu_thresh
    otsu_mask_mean = _image[np.where(otsu_mask)].mean()
    print("Otsu Mean: {}".format(otsu_mask_mean))
    brightness_thresh = max(image_otsu_thresh, otsu_mask_mean*0.8)
    print("Brightness Thresh: {}".format(brightness_thresh))
    
    # generate the canny edge image
    low_threshold = max(image_otsu_thresh * 0.66, 25)
    high_threshold = min(otsu_mask_mean, 200)
    print("[Canny] Low/High Threshold: {}/{}".format(low_threshold, high_threshold))
    canny_img = feature.canny(_image, sigma=0.8, 
                              low_threshold=low_threshold, 
                              high_threshold=high_threshold)
    edges = np.copy(canny_img)
    #plt.imshow(edges)
    #plt.show()
    
    
    # Detect two radii
    hough_radii = np.arange(radmin, radmax, radint)
    hough_res = hough_circle(edges, hough_radii)
    
    total_num_peaks = 30
    accum_min = 1.0
    while(accum_min > accum_thresh):
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=total_num_peaks)
        accum_min = accums[-1]
        total_num_peaks += 1
    
    print("Min Accum: {}, | Total Peaks: {}".format(accum_min, total_num_peaks))
        
    n_selected_circles = 0
    collection = []
    edge_noise_stack = []
    std_score_stack = []
    
    
    for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
        circle_i = (center_y, center_x, radius, accum_)
        #print("circle: {}".format(circle_i))
        if not circle_location_Pass(circle_i, _image, margin=margin):
            #print("location pass fail")
            continue
        if not watershed_dice_score_Pass(circle_i, _image, dice_thresh=dice_thresh):
            #print("dice score fail")
            continue
        if not circle_edge_noise_Pass(circle_i, edges, radDif=radDif, radVal=radVal, noise_thresh_=noise_thresh):
            #print("edge noise pass fail")
            continue
        if not circle_image_std_Pass(circle_i, _image, radDif=radDif, radVal=radVal, std_thresh_=std_thresh):
            #print("std score pass fail")
            continue
        if not circle_brightness_Pass(circle_i, _image, brightness_thresh):
            #print("brightness pass fail")
            continue
        if not overlap_Pass(circle_i, collection):
            #print("overlap fail")
            continue
        if not minDist_Pass(circle_i, collection, minDist):
            #print("minDist fail")
            continue
        collection.append(circle_i)
    
    n_selected_circles = len(collection)
    print("{} out of {} circles are selected.".format(n_selected_circles, total_num_peaks))
    
    
    if len(collection) > 0:
        cy = np.array([x[0] for x in collection[:n_circles]])
        cx = np.array([x[1] for x in collection[:n_circles]])
        radii = np.array([x[2] for x in collection[:n_circles]])
        accums = np.array([x[3] for x in collection[:n_circles]])
        brightness = np.array([circle_brightness_score(x, _image) for x in collection[:n_circles]])
        print("Brightness: {}".format(brightness))
        noises = np.array([circle_edge_noise_score(x, edges, radVal=5) for x in collection[:n_circles]])
        print("Noises: {}".format(noises))
        std_scores = np.array([circle_image_std_score(x, _image, radVal=5) for x in collection[:n_circles]])
        print("Std Scores: {}".format(std_scores))
        dice_scores = np.array([watershed_dice_score(x, _image) for x in collection[:n_circles]])
        print("Dice Scores: {}".format(dice_scores))
        
        if cy.max()>108 or cy.min()<20 or cx.max()>108 or cx.min()<20:
            print("#"*100,"Edge: ",cy,cx)
            
        if std_scores.max()>50:
            print("#"*100,"Std_Score: ",std_scores)
        
    else:
        cy, cx, radii, accums = np.array([]), np.array([]), np.array([]), np.array([])
        
    return accums, cx, cy, radii


def Ao_hough_circle1(image, radmin=8, radmax=15, radint=2, n_circles=2, minDist=10,
                     gaussian_sigma=1.2, radDif=None, radVal=None, margin=0.15,
                     accum_thresh=0.7, noise_thresh=20, std_thresh=40,
                     dice_thresh=0.7, wGamma=2.5):
    """
    Params
    ------
    image: input_image for finding circles
    radmin: min_radius
    radmax: max_radius
    radint: radius_interval
    n_circles: number_of_circles to be located
    minDist: minimum distance between each pair of circles

    Return
    ------
    """


    # apply gaussian filter, then rescale the image intensity
    _image = rescale_intensity(gaussian(image, sigma=gaussian_sigma), out_range=np.uint8).astype(np.uint8)

    # Image Otsu Threshold Mask Mean
    image_otsu_thresh = threshold_otsu(_image)
    print("Thresh: {}".format(image_otsu_thresh))
    otsu_mask = _image>image_otsu_thresh
    otsu_mask_mean = _image[np.where(otsu_mask)].mean()
    print("Otsu Mean: {}".format(otsu_mask_mean))
    brightness_thresh = max(image_otsu_thresh, otsu_mask_mean*0.8)

    # generate the canny edge image
    low_threshold = max(image_otsu_thresh * 0.66, 25)
    high_threshold = min(otsu_mask_mean, 200)
    print("[Canny] Low/High Threshold: {}/{}".format(low_threshold, high_threshold))
    canny_img = feature.canny(_image, sigma=0.8,
                              low_threshold=low_threshold,
                              high_threshold=high_threshold)
    edges = np.copy(canny_img)
    #plt.imshow(edges)
    #plt.show()


    # Detect two radii
    hough_radii = np.arange(radmin, radmax, radint)
    hough_res = hough_circle(edges, hough_radii)

    total_num_peaks = 30
    accum_min = 1.0
    while(accum_min > accum_thresh):
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                                   total_num_peaks=total_num_peaks)
        accum_min = accums[-1]
        total_num_peaks += 1

    print("Min Accum: {}, | Total Peaks: {}".format(accum_min, total_num_peaks))

    n_selected_circles = 0
    collection = []
    edge_noise_stack = []
    std_score_stack = []
    dice_score_stack = []

    #########################
    for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
        circle_i = (center_y, center_x, radius, accum_)
        dice_score = watershed_dice_score(circle_i, _image, gamma=wGamma)
        noise_score = circle_edge_noise_score(circle_i, edges, radDif, radVal)
        std_score = circle_image_std_score(circle_i, _image, radDif, radVal)
        dice_score_stack.append(dice_score)
        edge_noise_stack.append(noise_score)
        std_score_stack.append(std_score)


        collection.append(circle_i)

    dice_index = np.argsort(dice_score_stack)[::-1][:10]
    edge_index = np.argsort(edge_noise_stack)[:10]
    std_score_index = np.argsort(std_score_stack)[:10]

    overlap_index = [x for x in dice_index if x in edge_index]


    print("Dice Score: {}".format(np.array(dice_score_stack)[overlap_index]))
    collection2 = [collection[c_i] for c_i in overlap_index]
    #########################

    collection = []
    #for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
    for center_y, center_x, radius, accum_ in collection2:
        circle_i = (center_y, center_x, radius, accum_)
        #print("circle: {}".format(circle_i))
        if not circle_location_Pass(circle_i, _image, margin=margin):
            #print("location pass fail")
            continue
        #if not watershed_dice_score_Pass(circle_i, _image, dice_thresh=dice_thresh):
            #print("dice score fail")
        #    continue
        #if not circle_edge_noise_Pass(circle_i, edges, radDif=radDif, radVal=radVal, noise_thresh_=noise_thresh):
            #print("edge noise pass fail")
        #    continue
        if not circle_image_std_Pass(circle_i, _image, radDif=radDif, radVal=radVal, std_thresh_=std_thresh):
            #print("std score pass fail")
            continue
        if not circle_brightness_Pass(circle_i, _image, brightness_thresh):
            #print("brightness pass fail")
            continue
        if not overlap_Pass(circle_i, collection):
            #print("overlap fail")
            continue
        if not minDist_Pass(circle_i, collection, minDist):
            #print("minDist fail")
            continue
        collection.append(circle_i)

    n_selected_circles = len(collection)
    print("{} out of {} circles are selected.".format(n_selected_circles, total_num_peaks))


    if len(collection) > 0:
        cy = np.array([x[0] for x in collection[:n_circles]])
        cx = np.array([x[1] for x in collection[:n_circles]])
        radii = np.array([x[2] for x in collection[:n_circles]])
        accums = np.array([x[3] for x in collection[:n_circles]])
        brightness = np.array([circle_brightness_score(x, _image) for x in collection[:n_circles]])
        print("Brightness: {}".format(brightness))
        noises = np.array([circle_edge_noise_score(x, edges, radVal=5) for x in collection[:n_circles]])
        print("Noises: {}".format(noises))
        std_scores = np.array([circle_image_std_score(x, _image, radVal=5) for x in collection[:n_circles]])
        print("Std Scores: {}".format(std_scores))
        dice_scores = np.array([watershed_dice_score(x, _image) for x in collection[:n_circles]])
        print("Dice Scores: {}".format(dice_scores))

        if cy.max()>108 or cy.min()<20 or cx.max()>108 or cx.min()<20:
            print("#"*100,"Edge: ",cy,cx)

        if std_scores.max()>50:
            print("#"*100,"Std_Score: ",std_scores)

    else:
        cy, cx, radii, accums = np.array([]), np.array([]), np.array([]), np.array([])

    circles = list(zip(cy, cx, radii, accums))
    return circles

def Ao_hough_circleCV(image, n_circles=2, minDist=10,
                      param1=None, param2=30,
                      minRadius=6, maxRadius=14,
                      gaussian_sigma=0.8):

    _image = rescale_intensity(gaussian(image, sigma=gaussian_sigma), out_range=np.uint8).astype(np.uint8)

    if param1 is None:
        image_otsu_thresh = threshold_otsu(_image)
        otsu_mask = _image>image_otsu_thresh
        otsu_mask_mean = _image[np.where(otsu_mask)].mean()
        param1 = min(otsu_mask_mean, 200)

    circles_found = 0
    while(circles_found<n_circles):
        circles = cv2.HoughCircles(_image, cv2.HOUGH_GRADIENT, 1, minDist,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        circles_found = circles.shape[1] if circles is not None else 0
        param2 -= 1

    cx = [int(cc_[0]) for cc_ in circles[0]]
    cy = [int(cc_[1]) for cc_ in circles[0]]
    radii = [int(cc_[2]) for cc_ in circles[0]]
    accums = [0.5 for cc_ in circles[0]]
    circles = list(zip(cy, cx, radii, accums))[:n_circles]

    return circles

def Ao_hough_circle2(image, n_circles=2, minDist=10, minDist_CV=20,
                     param1=50, param2=30, minRadius=6, maxRadius=14,
                     gaussian_sigma=1.2, radDif=None, radVal=None, margin=0.15,
                     noise_thresh=20, std_thresh=40,
                     dice_thresh=0.7, wGamma=2.5):
    """
    Params
    ------
    image: input_image for finding circles
    n_circles: number_of_circles to be located
    minDist: minimum distance between each pair of circles

    Return
    ------
    """


    # apply gaussian filter, then rescale the image intensity
    _image = rescale_intensity(gaussian(image, sigma=gaussian_sigma), out_range=np.uint8).astype(np.uint8)

    # Image Otsu Threshold Mask Mean
    image_otsu_thresh = threshold_otsu(_image)
    print("Thresh: {}".format(image_otsu_thresh))
    otsu_mask = _image>image_otsu_thresh
    otsu_mask_mean = _image[np.where(otsu_mask)].mean()
    print("Otsu Mean: {}".format(otsu_mask_mean))
    brightness_thresh = max(image_otsu_thresh, otsu_mask_mean*0.8)

    # generate the canny edge image
    low_threshold = max(image_otsu_thresh * 0.66, 25)
    high_threshold = min(otsu_mask_mean, 200)
    print("[Canny] Low/High Threshold: {}/{}".format(low_threshold, high_threshold))
    canny_img = feature.canny(_image, sigma=0.8,
                              low_threshold=low_threshold,
                              high_threshold=high_threshold)
    edges = np.copy(canny_img)

    # Get the CV Circles
    circles_found = 0
    while(circles_found<10):
        circles = cv2.HoughCircles(_image, cv2.HOUGH_GRADIENT, 1, minDist_CV,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        circles_found = circles.shape[1] if circles is not None else 0
        param2 -= 1

    cx = [int(cc_[0]) for cc_ in circles[0]]
    cy = [int(cc_[1]) for cc_ in circles[0]]
    radii = [int(cc_[2]) for cc_ in circles[0]]
    accums = [0.5 for cc_ in circles[0]]
    # Get the CV Circles


    n_selected_circles = 0
    collection = []
    edge_noise_stack = []
    std_score_stack = []
    dice_score_stack = []

    #########################
    for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
        circle_i = (center_y, center_x, radius, accum_)
        dice_score = watershed_dice_score(circle_i, _image, gamma=wGamma)
        noise_score = circle_edge_noise_score(circle_i, edges, radDif, radVal)
        std_score = circle_image_std_score(circle_i, _image, radDif, radVal)
        dice_score_stack.append(dice_score)
        edge_noise_stack.append(noise_score)
        std_score_stack.append(std_score)


        collection.append(circle_i)

    dice_index = np.argsort(dice_score_stack)[::-1][:10]
    edge_index = np.argsort(edge_noise_stack)[:10]
    std_score_index = np.argsort(std_score_stack)[:10]

    overlap_index = [x for x in dice_index if x in edge_index]


    print("Dice Score: {}".format(np.array(dice_score_stack)[overlap_index]))
    collection2 = [collection[c_i] for c_i in overlap_index]
    #########################

    collection = []
    #for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
    for center_y, center_x, radius, accum_ in collection2:
        circle_i = (center_y, center_x, radius, accum_)
        #print("circle: {}".format(circle_i))
        if not circle_location_Pass(circle_i, _image, margin=margin):
            #print("location pass fail")
            continue
        #if not watershed_dice_score_Pass(circle_i, _image, dice_thresh=dice_thresh):
            #print("dice score fail")
        #    continue
        #if not circle_edge_noise_Pass(circle_i, edges, radDif=radDif, radVal=radVal, noise_thresh_=noise_thresh):
            #print("edge noise pass fail")
        #    continue
        if not circle_image_std_Pass(circle_i, _image, radDif=radDif, radVal=radVal, std_thresh_=std_thresh):
            #print("std score pass fail")
            continue
        if not circle_brightness_Pass(circle_i, _image, brightness_thresh):
            #print("brightness pass fail")
            continue
        if not overlap_Pass(circle_i, collection):
            #print("overlap fail")
            continue
        if not minDist_Pass(circle_i, collection, minDist):
            #print("minDist fail")
            continue
        collection.append(circle_i)

    n_selected_circles = len(collection)
    print("{} circles are selected.".format(n_selected_circles))


    if len(collection) > 0:
        cy = np.array([x[0] for x in collection[:n_circles]])
        cx = np.array([x[1] for x in collection[:n_circles]])
        radii = np.array([x[2] for x in collection[:n_circles]])
        accums = np.array([x[3] for x in collection[:n_circles]])
        brightness = np.array([circle_brightness_score(x, _image) for x in collection[:n_circles]])
        print("Brightness: {}".format(brightness))
        noises = np.array([circle_edge_noise_score(x, edges, radVal=5) for x in collection[:n_circles]])
        print("Noises: {}".format(noises))
        std_scores = np.array([circle_image_std_score(x, _image, radVal=5) for x in collection[:n_circles]])
        print("Std Scores: {}".format(std_scores))
        dice_scores = np.array([watershed_dice_score(x, _image) for x in collection[:n_circles]])
        print("Dice Scores: {}".format(dice_scores))

        if cy.max()>108 or cy.min()<20 or cx.max()>108 or cx.min()<20:
            print("#"*100,"Edge: ",cy,cx)

        if std_scores.max()>50:
            print("#"*100,"Std_Score: ",std_scores)

    else:
        cy, cx, radii, accums = np.array([]), np.array([]), np.array([]), np.array([])

    circles = list(zip(cy, cx, radii, accums))
    return circles

def Ao_hough_circle3(image, n_circles=2, minDist=10, minDist_CV=20,
                     param1=50, param2=30, minRadius=6, maxRadius=14,
                     gaussian_sigma=1.2, radDif=None, radVal=None, margin=0.15,
                     noise_thresh=20, std_thresh=40,
                     dice_thresh=0.7, wGamma=2.5):
    """
    Params
    ------
    image: input_image for finding circles
    n_circles: number_of_circles to be located
    minDist: minimum distance between each pair of circles

    Return
    ------
    """


    # apply gaussian filter, then rescale the image intensity
    _image = rescale_intensity(gaussian(image, sigma=gaussian_sigma), out_range=np.uint8).astype(np.uint8)

    # Image Otsu Threshold Mask Mean
    image_otsu_thresh = threshold_otsu(_image)
    print("Thresh: {}".format(image_otsu_thresh))
    otsu_mask = _image>image_otsu_thresh
    otsu_mask_mean = _image[np.where(otsu_mask)].mean()
    print("Otsu Mean: {}".format(otsu_mask_mean))
    brightness_thresh = max(image_otsu_thresh, otsu_mask_mean*0.8)

    # generate the canny edge image
    low_threshold = max(image_otsu_thresh * 0.66, 25)
    high_threshold = min(otsu_mask_mean, 200)
    print("[Canny] Low/High Threshold: {}/{}".format(low_threshold, high_threshold))
    canny_img = feature.canny(_image, sigma=0.8,
                              low_threshold=low_threshold,
                              high_threshold=high_threshold)
    edges = np.copy(canny_img)

    # Get the CV Circles
    circles_found = 0
    while(circles_found<10):
        circles = cv2.HoughCircles(_image, cv2.HOUGH_GRADIENT, 1, minDist_CV,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)
        circles_found = circles.shape[1] if circles is not None else 0
        param2 -= 1

    cx = [int(cc_[0]) for cc_ in circles[0]]
    cy = [int(cc_[1]) for cc_ in circles[0]]
    radii = [int(cc_[2]) for cc_ in circles[0]]
    accums = [0.5 for cc_ in circles[0]]
    # Get the CV Circles


    collection = []
    for center_y, center_x, radius, accum_ in zip(cy, cx, radii, accums):
        circle_i = (center_y, center_x, radius, accum_)
        #print("circle: {}".format(circle_i))
        if not circle_location_Pass(circle_i, _image, margin=margin):
            print("location pass fail")
            continue
        if not watershed_dice_score_Pass(circle_i, _image, dice_thresh=dice_thresh):
            print("dice score fail")
            continue
        if not circle_edge_noise_Pass(circle_i, edges, radDif=radDif, radVal=radVal, noise_thresh_=noise_thresh):
            print("edge noise pass fail")
            continue
        if not circle_image_std_Pass(circle_i, _image, radDif=radDif, radVal=radVal, std_thresh_=std_thresh):
            print("std score pass fail")
            continue
        if not circle_brightness_Pass(circle_i, _image, brightness_thresh):
            print("brightness pass fail")
            continue
        if not overlap_Pass(circle_i, collection):
            print("overlap fail")
            continue
        if not minDist_Pass(circle_i, collection, minDist):
            print("minDist fail")
            continue
        collection.append(circle_i)

    n_selected_circles = len(collection)
    print("{} circles are selected.".format(n_selected_circles))


    if len(collection) > 0:
        cy = np.array([x[0] for x in collection[:n_circles]])
        cx = np.array([x[1] for x in collection[:n_circles]])
        radii = np.array([x[2] for x in collection[:n_circles]])
        accums = np.array([x[3] for x in collection[:n_circles]])
        brightness = np.array([circle_brightness_score(x, _image) for x in collection[:n_circles]])
        print("Brightness: {}".format(brightness))
        noises = np.array([circle_edge_noise_score(x, edges, radVal=5) for x in collection[:n_circles]])
        print("Noises: {}".format(noises))
        std_scores = np.array([circle_image_std_score(x, _image, radVal=5) for x in collection[:n_circles]])
        print("Std Scores: {}".format(std_scores))
        dice_scores = np.array([watershed_dice_score(x, _image) for x in collection[:n_circles]])
        print("Dice Scores: {}".format(dice_scores))

        if cy.max()>108 or cy.min()<20 or cx.max()>108 or cx.min()<20:
            print("#"*100,"Edge: ",cy,cx)

        if std_scores.max()>50:
            print("#"*100,"Std_Score: ",std_scores)

    else:
        cy, cx, radii, accums = np.array([]), np.array([]), np.array([]), np.array([])

    circles = list(zip(cy, cx, radii, accums))
    return circles
