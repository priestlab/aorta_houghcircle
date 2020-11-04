import cv2
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from multiprocessing import Pool

from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.io import imsave

from segmentation.Ao_circle import Ao_hough_circle1
from segmentation.Ao_circle import Ao_hough_circleCV
from segmentation.watershed import watershed_all_circles

from utils.utils import organize_circles
from utils.display import get_watershed_overlay
from utils.display import get_overlay_circles
from utils.file import secure_dir

from dataset.AoDist_dataset import AoDistDataset



def segment_frame(frame, plot=False, savePng=None):
    """
    Segmenting one frame for ascending and descending aorta
    """
    _image = rescale_intensity(gaussian(frame, sigma=0.8), out_range=np.uint8).astype(np.uint8)
    #print(f"IMAGE ID: {image_id}")
    
    ################################################
    ## Get the circles
    ################################################
    circles = Ao_hough_circle1(_image, radmin=6, radmax=16, radint=1, n_circles=2, minDist=10,
                               gaussian_sigma=0.8, noise_thresh=5, std_thresh=20,
                               accum_thresh=0.5, radDif=-2, radVal=None, margin=0.15,
                               dice_thresh=0.8)
    if (len(circles)<2):
        print("-"*77)
        print("Using CV Hough Circle")
        circles = Ao_hough_circleCV(_image, n_circles=2, minDist=20, gaussian_sigma=0.8,
                                    param1=100, param2=30, minRadius=6, maxRadius=16)
        #circles = Ao_hough_circle3(image, n_circles=2, minDist=10, minDist_CV=20,
        #                           param1=50, param2=30, minRadius=6, maxRadius=16,
        #                           gaussian_sigma=0.8, noise_thresh=5, std_thresh=20,
        #                           dice_thresh=0.8, radDif=-2, radVal=None, margin=0.15,
        #                           wGamma=2.5)
    
    circles = organize_circles(circles)
    circle_overlay_img = get_overlay_circles(_image, circles)
    
    
    ################################################
    ## Get the segmentations and markers
    ################################################
    labelValue = [50, 100, 150]
    c_seg, c_marker = watershed_all_circles(_image, circles, labelValue, gamma=2.5)
    c_overlay = get_watershed_overlay(_image, c_seg)
    
    if plot:
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(c_overlay)
        ax1.set_title("segmentation")
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(c_seg==50)
        ax2.set_title("ascending aorta mask")
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(c_seg==100)
        ax3.set_title("descending aorta mask")
        plt.show()

    if savePng is not None:
        secure_dir(savePng)
        imsave(f"{savePng}/image.png", frame)
        imsave(f"{savePng}/masked.png", c_overlay)

    
    return c_seg

def get_aorta_measurements(c_seg, image_ratio, label):
    """
    Calculating aorta measurements using the segmentation mask, image_ratio, and label
    - c_seg : segmentation mask
    - image_ratio : ratio for translating pixel length to actual length in mm
    - label : the label (e.g. 50, 100) used in c_seg for aorta mask
    """
    aorta_area = cv2.dilate((c_seg==label).astype(np.uint8), None, iterations=1).sum()
    aorta_measurement = ((aorta_area / math.pi)**0.5)*2.0*image_ratio
    return aorta_measurement

def first_frame_segment(data, save_dir="aorta", plot=False, savePng=False):
    """
    Generate ascending and descending aorta diameters using first frame
    """
    images, image_ratio, image_id = data
    image = images[0]

    if savePng:
        c_seg = segment_frame(image, plot=plot, savePng=f"{save_dir}/png/{image_id}_f0")
    else:
        c_seg = segment_frame(image, plot=plot)

    aorta_measurements = np.array([get_aorta_measurements(c_seg, image_ratio, 50),
                                   get_aorta_measurements(c_seg, image_ratio, 100)])

    print(f"aorta measurement in {image_id}: {aorta_measurements}")
    if save_dir is not None:
        np.save(f"{save_dir}/npy/{image_id}.npy", aorta_measurements)

    return aorta_measurements


def all_frames_segment(data, save_dir="aorta", plot=False, savePng=False):
    """
    Generate ascending and descending aorta diameters using all frames
    """
    images, image_ratio, image_id = data


    if savePng:
        c_segs = np.array([segment_frame(image, plot=plot, savePng=f"{save_dir}/png/{image_id}_f{i}") for i, image in enumerate(images)])
    else:
        c_segs = np.array([segment_frame(image, plot=plot) for image in images])

    c_seg1 = np.mean((c_segs==50).astype(float), axis=0).round().astype(int)
    c_seg2 = np.mean((c_segs==100).astype(float), axis=0).round().astype(int)

    aorta_measurements = np.array([get_aorta_measurements(c_seg1, image_ratio, 1),
                                   get_aorta_measurements(c_seg2, image_ratio, 1)])

    print(f"aorta measurement in {image_id}: {aorta_measurements}")
    if save_dir is not None:
        np.save(f"{save_dir}/npy/{image_id}.npy", aorta_measurements)

    return aorta_measurements

def segment_dtpt(data, save_dir="aorta", all_frames=False, plot=False, savePng=False):
    image_id = data[-1]
    print("-"*100)
    print(f"SEGMENTING {image_id}")
    try:
        if all_frames:
            aorta_measurements = all_frames_segment(data, save_dir, plot=plot, savePng=savePng)
        else:
            aorta_measurements = first_frame_segment(data, save_dir, plot=plot, savePng=savePng)


    except Exception as err:
        print("Caught an error in {}".format(image_id))
        print("Exception: {}".format(err))
        aorta_measurements = None

    return aorta_measurements, image_id


def main(args):
    secure_dir(f"{args.out_dir}/npy")

    if args.all_frames:
        print("Using all frames to calculating aorta measurements.")
    
    scale_size = (196, 196)
    data_size = (128, 128)
    seriesDescription = "CINE_segmented_Ao_dist"
    meta = ["PixelSpacing", "Rows", "Columns"]

    dataset = AoDistDataset(args.root_dir, args.csv_data, 
                            scale_size, data_size, 
                            seriesDescription, meta)

    segment_dt = partial(segment_dtpt, save_dir=args.out_dir, all_frames=args.all_frames, 
                         plot=args.plot, savePng=args.save_png)

    #results = []
    #for i, data in enumerate(dataset):
    #    print(f"Processing No.{i} datapoint...")
    #    results.append(segment_dt(data))
    pool = Pool(args.threads)
    results = pool.map(segment_dt, dataset)

    df = pd.DataFrame(columns=['PID', "AscendingAorta", "DescendingAorta"])
    for result in results:
        aorta_measurements, image_id = result
        if aorta_measurements is not None:
            df = df.append({"PID": image_id,
                            "AscendingAorta": aorta_measurements[0],
                            "DescendingAorta": aorta_measurements[1]},
                            ignore_index=True)

    df.to_csv(f"{args.out_dir}/aorta_measurements.csv", index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-R", "--root_dir", type=str, required=True, help="the folder to read data from.")
    argparser.add_argument("-C", "--csv_data", type=str, default="labels.csv", help="the csv file to load image path/pid from.")
    argparser.add_argument("-O", "--out_dir", type=str, default="aorta", help="the output folder to save csv and npy measurements to.")
    argparser.add_argument("--threads", type=int, default=8, help="the number of threads to use.")
    argparser.add_argument("--all_frames", action="store_true", help="whether to use all frames for calculating aorta measurements.")
    argparser.add_argument("--plot", action="store_true", help="whether to plot the segmentation.")
    argparser.add_argument("--save_png", action="store_true", help="whether to save the segmentation images into pngs.")
    args = argparser.parse_args()

    main(args)





