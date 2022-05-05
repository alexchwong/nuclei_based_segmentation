"""Utility functions for the HPA Cell Segmentation package."""
import os.path
import urllib
import zipfile

import pandas as pd
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)
import cv2
import imageio
from tqdm import tqdm

HIGH_THRESHOLD = 0.4
LOW_THRESHOLD = HIGH_THRESHOLD - 0.25


def download_with_url(url_string, file_path, unzip=False):
    """Download file with a link."""
    with urllib.request.urlopen(url_string) as response, open(
        file_path, "wb"
    ) as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    if unzip:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(file_path))

def facs_two_images(segmentation_mask, image1, image2, label1 = "marker1", label2 = "marker2"):
    '''
        Function to generate a FACS-like data frame of results
        inputs:
        * segmentation_mask: the value returned by segment_basic()
        * image1, image2: images of individual channels
        * label1, label2: labels of individual channels
        outputs:
        * a data frame containing:
          * label: cell ID
          * area: pixel area of cell
          * diameter: diameter of cell
          * perimeter: perimeter of cell
          * xmin, ymin, xmax, ymax: bounding box of cell
          * label1, label2: mean channel intensity of cell    
    '''
    def _preprocess_img_full(image):
        if isinstance(image, str):
            image = imageio.imread(image)
        return image
    image1 = _preprocess_img_full(image1)
    image2 = _preprocess_img_full(image2)
    
    props1 = measure.regionprops_table(segmentation_mask, 
        cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY),
        properties=['label', 'area', 'equivalent_diameter', 'bbox',
            'mean_intensity', 'solidity', 'orientation', 'perimeter'])
    props2 = measure.regionprops_table(segmentation_mask, 
        cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY),
        properties=['label', 'area', 'equivalent_diameter',
            'mean_intensity', 'solidity', 'orientation', 'perimeter'])
    out1 = pd.DataFrame(props1).iloc[1:,:]
    out2 = pd.DataFrame(props2).iloc[1:,:]
    final = pd.concat([out1.label, out1.area, out1.equivalent_diameter, out1.perimeter,
        out1['bbox-0'], out1['bbox-1'], out1['bbox-2'], out1['bbox-3'],
        out1.mean_intensity, out2.mean_intensity], axis = 1,
        keys = ["cell_ID", "area", "diameter", "perimeter", 
            "xmin", "ymin", "xmax", "ymax",
            label1, label2])
    return final

def facs_all_channels(segmentation_mask, image_list, label_list):
    '''
        Function to generate a FACS-like data frame of results
        inputs:
        * segmentation_mask: the value returned by segment_basic()
        * image1, image2: images of individual channels
        * label1, label2: labels of individual channels
        outputs:
        * a data frame containing:
          * label: cell ID
          * area: pixel area of cell
          * diameter: diameter of cell
          * perimeter: perimeter of cell
          * xmin, ymin, xmax, ymax: bounding box of cell
          * label1, label2: mean channel intensity of cell    
    '''

    def _preprocess_img_full(image):
        if isinstance(image, str):
            image = imageio.imread(image)
        return image
        
    preprocessed_imgs = []
    for img in tqdm(image_list):
        preprocessed_imgs.append(_preprocess_img_full(img))
    
    for index, (image, label) in enumerate(zip(preprocessed_imgs, label_list)):
        if index == 0:
            props = measure.regionprops_table(segmentation_mask, 
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                properties=['label', 'area', 'equivalent_diameter', 'bbox', 'perimeter', 'mean_intensity'])
        else:
            props = measure.regionprops_table(segmentation_mask, 
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                properties=['mean_intensity'])
        out = pd.DataFrame(props)
        
        if index == 0:
            final = pd.concat([out.label, out.area, out.equivalent_diameter, out.perimeter,
                out['bbox-0'], out['bbox-1'], out['bbox-2'], out['bbox-3'],
                out.mean_intensity], axis = 1,
                keys = ["cell_ID", "area", "diameter", "perimeter", 
                    "xmin", "ymin", "xmax", "ymax", label])
        else:
            final = pd.concat([final, out.mean_intensity], axis = 1)
            final.columns = [*final.columns[:-1], label]
    
    return final