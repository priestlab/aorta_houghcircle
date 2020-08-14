from __future__ import division

import numpy as np

from pydicom import dcmread
from pydicom.filebase import DicomBytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset
from skimage.exposure import rescale_intensity
import pandas as pd
from scipy.ndimage import center_of_mass

from dataset.processing import CropResize
from dataset.processing import ResizePad

def load_dcm_from_zip(zip_file, seriesDescription=None, meta=None):

    ## load in all the dcms from the input zip file
    zfile = ZipFile(zip_file)
    content = zfile.namelist()
    dcms = [dcmread(DicomBytesIO(zfile.read(x)))
            for x in content if x.endswith('.dcm')]

    ## filter the dcms by the SeriesDescription
    if seriesDescription is not None:
        dcms = [x for x in dcms 
                if x.SeriesDescription == seriesDescription]

    ## filter the dcms by the first SeriesInstanceUID in each SeriesDescription
    series_pairs = {}
    for dcm in dcms:
        if dcm.SeriesDescription in series_pairs.keys():
            if dcm.SeriesInstanceUID not in series_pairs[dcm.SeriesDescription]:
                series_pairs[dcm.SeriesDescription].append(dcm.SeriesInstanceUID)
        else:
            series_pairs[dcm.SeriesDescription] = [dcm.SeriesInstanceUID]

    seriesInstanceUIDs = [sorted(x)[0] for x in series_pairs.values()]
    dcms = [x for x in dcms 
            if x.SeriesInstanceUID in seriesInstanceUIDs]
    
    ## sort the dcms by SeriesInstanceUID then InstanceNumber
    dcms = sorted(dcms, key = lambda x: (x.SeriesInstanceUID, x.InstanceNumber))

    meta_keys = ['SeriesDescription', 'SeriesInstanceUID', 'InstanceNumber']
    if meta is not None:
        meta_keys += meta
    npys = np.array([x.pixel_array for x in dcms])
    metas = [dict([(key, getattr(x, key)) for key in meta_keys]) for x in dcms]
        

    return npys, metas



##################################################
## resize to (196, 196), then crop down 
## to (128, 128), so lose 34 pixels on each side.
##################################################
def preprocess_image(image,
                     scale_size=(196,196),
                     data_size=(128,128)):
    """
    - rescale image intensity to uint8
    - crop the image to square shape
        then resize to scale_size
    - crop the edges to be data_size
    
    Params
    ------
    - image
    - scale_size
    - data_size
    
    Return
    ------
    - image
    
    """
    
    image = rescale_intensity(image, out_range=np.uint8).astype(np.uint8)
    image_scaled = CropResize(scale_size).crop_resize_image(image)

    cy, cx = center_of_mass(image_scaled)
    crop = (int(min(max(cy-data_size[0]/2,0),scale_size[0]-data_size[0])),
            int(min(max(cx-data_size[1]/2,0),scale_size[1]-data_size[1])))
    image_cropped = image_scaled[crop[0]:crop[0]+data_size[0],
                                 crop[1]:crop[1]+data_size[1]]
    paddings = [(crop[0], scale_size[0]-data_size[0]-crop[0]),
                (crop[1], scale_size[1]-data_size[1]-crop[1]),
                image.shape]
    
    return image_cropped, paddings


def postprocess_mask(mask, paddings, image_shape):
    padded_mask = np.pad(mask, paddings, 'constant')
    scaled_mask = ResizePad(padded_mask, tuple(image_shape))

    return scaled_mask
            
            
##################################################
## AoDistDataset for loading Ao_Dist data
##################################################
class AoDistDataset(Dataset):
    """
    Dataset for Ao_dist MRI data
    scale to scale_size
    crop down to data_size
    """
    def __init__(self, root_dir, csv_data, 
                 scale_size=(196,196),
                 data_size=(128,128),
                 seriesDescription="CINE_segmented_Ao_dist",
                 meta=["PixelSpacing", "Rows", "Columns"],
                 return_paddings=False):
        """
        Params
        ------
        - root_dir
        - csv_data
        - scale_size
        - data_size
        """
        self.root_dir = root_dir
        self.csv_data = csv_data
        self.scale_size = scale_size
        self.data_size = data_size
        #self.ImageCropResize = CropResize(data_size)
        
        self.labels = pd.read_csv(csv_data) if type(csv_data) is str else csv_data

        self.seriesDescription = seriesDescription
        self.meta = meta
        self.return_paddings = return_paddings
        
    def __len__(self):
        return len(self.labels)
    
    def __getitemOLD__(self, idx):
        image_path = self.labels.iloc[idx, 0]
        image_pid = image_path.split("/")[0]
        image = np.load("{}/{}".format(self.root_dir, image_path))
        image = preprocess_image(image, self.scale_size, self.data_size)
        
        return image, image_pid
    
    def __getitem__(self, idx):
        zip_path = self.labels.iloc[idx, 0]
        npys, metas = load_dcm_from_zip("{}/{}".format(self.root_dir, zip_path), 
                                        self.seriesDescription, self.meta)
        ratio = float(min(npys.shape[1], npys.shape[2])) / self.scale_size[0]
        ratio = ratio * metas[0]['PixelSpacing'][0]
        
        image_pid = zip_path.split("/")[-1].split(".")[0].split("_")[0]
        images_w_paddings = [preprocess_image(image, self.scale_size, self.data_size) 
                             for image in npys]
        images = np.array([x[0] for x in images_w_paddings])
        paddings = np.array([x[1] for x in images_w_paddings])
        
        if self.return_paddings:
            return images, npys, paddings, ratio, image_pid
        else:
            return images, ratio, image_pid
    


