"""Utility functions for the HPA Cell Segmentation package."""
import os.path
import urllib
import zipfile

import numpy as np
import scipy.ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)

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


def label_nuclei(nuclei_pred, cell_image,
                 NUCLEUS_THRESHOLD = 0.4,
                 BORDER_THRESHOLD = 0.15,
                 IMAGE_THRESHOLD = 0.15,
                 nuc_small_obj_size = 10, 
                 cell_small_obj_size = 20,
                 cell_small_hole_size = 5):
    """Return the labeled nuclei mask data array.
    This function works best for Human Protein Atlas cell images with
    predictions from the CellSegmentator class.
    Keyword arguments:
    nuclei_pred -- a 3D numpy array of a prediction from a nuclei image.
    cell_image -- a 3D numpy array of combined channel image (BGR - called by cv2.imread())
    NUCLEUS_THRESHOLD (0.4): minimum intensity (0-1) of blue channel (in nuclei_pred) to be considered positive staining
    BORDER_THRESHOLD (0.15): minimum intensity of green channel (in nuclei_pred) to be considered cell border
    IMAGE_THRESHOLD (0.15): minimum intensity of greyscaled cell_image to be considered part of a cell (depends on exposure)
    nuc_small_obj_size (10): minimum size to be considered nucleus
    cell_small_obj_size (20): minimum size to be considered a cell
    cell_small_hole_size (5): remove holes of this size or less
    Returns:
    cell-label -- An array with unique numbers for each found cell
                    in the nuclei_pred. A value of 0 in the array is
                    considered background, and the values 1-n is the
                    areas of the cells 1-n.
    """
    img_copy = np.copy(nuclei_pred[..., 2])
    borders = (nuclei_pred[..., 1] > BORDER_THRESHOLD * 256).astype(np.uint8)
    m = img_copy * (1 - borders)

    img_copy[m <= NUCLEUS_THRESHOLD * 256] = 0
    img_copy[m > NUCLEUS_THRESHOLD * 256] = 1
    img_copy = img_copy.astype(bool)
    img_copy = binary_erosion(img_copy)
    # TODO: Add parameter for remove small object size for
    #       differently scaled images.
    img_copy = remove_small_objects(img_copy, nuc_small_obj_size)
    img_copy = img_copy.astype(np.uint8)
    markers = measure.label(img_copy).astype(np.uint32)

    mask_img = np.copy(cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY))
    mask_img[mask_img <= IMAGE_THRESHOLD * 256] = 0
    mask_img[mask_img > IMAGE_THRESHOLD * 256] = 1
    mask_img = mask_img.astype(bool)
    
    mask_img = remove_small_objects(mask_img, cell_small_obj_size)
    mask_img = remove_small_holes(mask_img, cell_small_hole_size)
    
    # TODO: Figure out good value for remove small objects.
    
    mask_img = mask_img.astype(np.uint8)
    nuclei_label = segmentation.watershed(
        mask_img, markers, mask=mask_img, watershed_line=True
    )
    # nuclei_label = remove_small_objects(nuclei_label, 2500)
    
    nuclei_label = measure.label(nuclei_label)
    return nuclei_label

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
    
    props1 = measure.regionprops_table(segmentation_mask, 
        cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY),
        properties=['label', 'area', 'equivalent_diameter', 'bbox',
            'mean_intensity', 'solidity', 'orientation', 'perimeter'])
    props2 = measure.regionprops_table(segmentation_mask, 
        cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY),
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
    
    for index, (image, label) in enumerate(zip(image_list, label_list)):
        if index == 0:
            props = measure.regionprops_table(segmentation_mask, 
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                properties=['label', 'area', 'equivalent_diameter', 'bbox', 'perimeter', 'mean_intensity'])
        else:
            props = measure.regionprops_table(segmentation_mask, 
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
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