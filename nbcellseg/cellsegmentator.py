"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.functional as F
from nbcellseg.constants import (MULTI_CHANNEL_CELL_MODEL_URL,
                                  NUCLEI_MODEL_URL, TWO_CHANNEL_CELL_MODEL_URL)
from nbcellseg.utils import download_with_url
from skimage import transform, util

from skimage import filters, measure, segmentation
from skimage.morphology import (binary_erosion, closing, disk,
                                remove_small_holes, remove_small_objects)

from tqdm import tqdm

NORMALIZE = {"mean": [124 / 255, 117 / 255, 104 / 255], "std": [1 / (0.0167 * 255)] * 3}


class CellSegmentator(object):
    """Uses pretrained DPN-Unet models to segment cells from images."""

    def __init__(
        self,
        nuclei_model="./nuclei_model.pth",
        scale_factor=0.25,
        device="cuda",
        padding=False,
    ):
        """Class for segmenting nuclei and whole cells from confocal microscopy images.

        It takes lists of images and returns the raw output from the
        specified segmentation model. Models can be automatically
        downloaded if they are not already available on the system.

        When working with images from the Huan Protein Cell atlas, the
        outputs from this class' methods are well combined with the
        label functions in the utils module.

        Note that for cell segmentation, there are two possible models
        available. One that works with 2 channeled images and one that
        takes 3 channels.

        Keyword arguments:
        nuclei_model -- A loaded torch nuclei segmentation model or the
                        path to a file which contains such a model.
                        If the argument is a path that points to a non-existant file,
                        a pretrained nuclei_model is going to get downloaded to the
                        specified path (default: './nuclei_model.pth').
        scale_factor -- How much to scale images before they are fed to
                        segmentation models. Segmentations will be scaled back
                        up by 1/scale_factor to match the original image
                        (default: 0.25).
        device -- The device on which to run the models.
                  This should either be 'cpu' or 'cuda' or pointed cuda
                  device like 'cuda:0' (default: 'cuda').
        padding -- Whether to add padding to the images before feeding the
                   images to the network. (default: False).
        
        Exported functions:
          - pred_nuclei(images): Creates 2-channel (-/G/B) numpy array from model
              output. The green channel represent cell borders; the blue channel
              represent cell body.
          - label_cells(pred, host_image): Use nuclei model prediction and the
              original fluorescent image to create segmentation markers. If the
              `host_image` is the DAPI channel, this outputs nuclei mask. If
              `host_image` is combined channels, this outputs cell mask.
          - get_cell_markers: Gets cell markers from a list of DAPI and combined channel images
        """
        if device != "cuda" and device != "cpu" and "cuda" not in device:
            raise ValueError(f"{device} is not a valid device (cuda/cpu)")
        if device != "cpu":
            try:
                assert torch.cuda.is_available()
            except AssertionError:
                print("No GPU found, using CPU.", file=sys.stderr)
                device = "cpu"
        self.device = device

        if isinstance(nuclei_model, str):
            if not os.path.exists(nuclei_model):
                print(
                    f"Could not find {nuclei_model}. Downloading it now",
                    file=sys.stderr,
                )
                download_with_url(NUCLEI_MODEL_URL, nuclei_model)
            nuclei_model = torch.load(
                nuclei_model, map_location=torch.device(self.device)
            )
        if isinstance(nuclei_model, torch.nn.DataParallel) and device == "cpu":
            nuclei_model = nuclei_model.module

        self.nuclei_model = nuclei_model.to(self.device)
        self.scale_factor = scale_factor
        self.padding = padding

    def pred_nuclei(self, images):
        """Predict the nuclei segmentation.

        Keyword arguments:
        images -- A list of image arrays or a list of paths to images.
                  If as a list of image arrays, the images could be 2d images
                  of nuclei data array only, or must have the nuclei data in
                  the blue channel; If as a list of file paths, the images
                  could be RGB image files or gray scale nuclei image file
                  paths.

        Returns:
        predictions -- A list of predictions of nuclei segmentation for each nuclei image.
        """

        def _preprocess(image):
            if isinstance(image, str):
                image = imageio.imread(image)
            self.target_shape = image.shape
            if len(image.shape) == 2:
                image = np.dstack((image, image, image))
            image = transform.rescale(image, self.scale_factor, multichannel=True)
            nuc_image = np.dstack((image[..., 2], image[..., 2], image[..., 2]))
            if self.padding:
                rows, cols = nuc_image.shape[:2]
                self.scaled_shape = rows, cols
                nuc_image = cv2.copyMakeBorder(
                    nuc_image,
                    32,
                    (32 - rows % 32),
                    32,
                    (32 - cols % 32),
                    cv2.BORDER_REFLECT,
                )
            nuc_image = nuc_image.transpose([2, 0, 1])
            return nuc_image

        def _segment_helper(imgs):
            with torch.no_grad():
                mean = torch.as_tensor(NORMALIZE["mean"], device=self.device)
                std = torch.as_tensor(NORMALIZE["std"], device=self.device)
                imgs = torch.tensor(imgs).float()
                imgs = imgs.to(self.device)
                imgs = imgs.sub_(mean[:, None, None]).div_(std[:, None, None])

                imgs = self.nuclei_model(imgs)
                imgs = F.softmax(imgs, dim=1)
                return imgs

        preprocessed_imgs = map(_preprocess, images)
        predictions = map(lambda x: _segment_helper([x]), preprocessed_imgs)
        predictions = map(lambda x: x.to("cpu").numpy()[0], predictions)
        predictions = map(util.img_as_ubyte, predictions)
        predictions = list(tqdm(map(self._restore_scaling_padding, predictions)))
        return predictions

    def _restore_scaling_padding(self, n_prediction):
        """Restore an image from scaling and padding.

        This method is intended for internal use.
        It takes the output from the nuclei model as input.
        """
        n_prediction = n_prediction.transpose([1, 2, 0])
        if self.padding:
            n_prediction = n_prediction[
                32 : 32 + self.scaled_shape[0], 32 : 32 + self.scaled_shape[1], ...
            ]
        n_prediction[..., 0] = 0
        if not self.scale_factor == 1:
            n_prediction = cv2.resize(
                n_prediction,
                (self.target_shape[0], self.target_shape[1]),
                interpolation=cv2.INTER_AREA,
            )
        return n_prediction

    def label_cells(self, nuclei_pred, host_image,
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
        host_image -- a 3D numpy array (RGB) or image path of fluorescent image
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

        if isinstance(host_image, str):
            host_image = imageio.imread(host_image)
        mask_img = np.copy(cv2.cvtColor(host_image, cv2.COLOR_RGB2GRAY))
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

    def get_cell_markers(self,
        images_dapi, images_combined,
        NUCLEUS_THRESHOLD = 0.4,
        BORDER_THRESHOLD = 0.15,
        threshold_list = np.arange(0.05, 0.2, 0.02),
        nuc_small_obj_size = 10, 
        cell_small_obj_size = 20,
        cell_small_hole_size = 5):
        '''
            Function to return segmentation markers
            inputs:
            * images_dapi: DAPI channel images. May be a numpy array (RGB) or a list; may be file names
                DAPI staning is presumed to be in the blue channel
            * images_combined: Combined (all) channel images.
                May be a numpy array (RGB) or a list; may be file names
            * NUCLEUS_THRESHOLD: Fraction blue channel to define nuclei body
            * BORDER_THRESHOLD: Fraction green channel to define edges between cells
            * nuc_small_obj_size: Nuclei predictions smaller than this is removed
            * cell_small_obj_size: cells smaller than this will be removed
            * cell_small_hole_size: cell holes smaller than this will be removed
        '''
        def _label_nuclei_custom(pred, img, intensity):
            return self.label_cells(pred, img,
                NUCLEUS_THRESHOLD = NUCLEUS_THRESHOLD,
                BORDER_THRESHOLD = BORDER_THRESHOLD,
                IMAGE_THRESHOLD = intensity,
                nuc_small_obj_size = nuc_small_obj_size, 
                cell_small_obj_size = cell_small_obj_size,
                cell_small_hole_size = cell_small_hole_size)
            
        def _get_stats_from_mask(mask):
            props = measure.regionprops_table(mask, properties=['area'])
            out = pd.DataFrame(props)
            return out.area.mean(), out.area.std()

        def _determine_image_threshold(nuclei_pred, cell_image, threshold_list):
            nuclei_masks = [_label_nuclei_custom(nuclei_pred, cell_image, x) for x in threshold_list]
            nstds = []
            for mask in nuclei_masks:
                mask_mean, mask_std = _get_stats_from_mask(mask)
                nstds.append(mask_std/mask_mean)
            
            min_nstd = min(nstds)
            min_index = nstds.index(min_nstd)
            return round(threshold_list[min_index], 2)
        
        def _get_optimum_markers(pred, img, threshold_list):
            opt_img_threshold = _determine_image_threshold(pred, img, threshold_list)
            return _label_nuclei_custom(pred, img, opt_img_threshold)

        def _preprocess_img_full(image):
            if isinstance(image, str):
                image = imageio.imread(image)
            return image

        print("Loading images")
        if not isinstance(images_combined, list):
            images_combined = [images_combined]
            
        preprocessed_imgs = []
        for img in tqdm(images_combined):
            preprocessed_imgs.append(_preprocess_img_full(img))
        
        print("Detecting nuclei")
        nuc_segmentations = self.pred_nuclei(images_dapi)
        print("Optimizing segmentations")
        markers = []
        for pred, img in tqdm(zip(nuc_segmentations, preprocessed_imgs)):
            markers.append(_get_optimum_markers(pred, img, threshold_list))

        return markers, nuc_segmentations