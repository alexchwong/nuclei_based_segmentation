"""Package for loading and running the nuclei and cell segmentation models programmaticly."""
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from nbcellseg.constants import (MULTI_CHANNEL_CELL_MODEL_URL,
                                  NUCLEI_MODEL_URL, TWO_CHANNEL_CELL_MODEL_URL)
from nbcellseg.utils import download_with_url, label_nuclei
from skimage import transform, util

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
        predictions = list(map(self._restore_scaling_padding, predictions))
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

    def label_cells(self, 
        img_nuclear, img_full, 
        NUCLEUS_THRESHOLD = 0.4,
        BORDER_THRESHOLD = 0.15,
        IMAGE_THRESHOLD = 0.12,
        nuc_small_obj_size = 10, 
        cell_small_obj_size = 20,
        cell_small_hole_size = 5):
        '''
            Function to call the segmentation algorithm
            inputs:
            * img_nuclear: DAPI image (input of cv.imread() in BGR format)
            * img_full: combined cell image (cv.imread format)
            * cell_image_intensity_threshold (0-1): when img_full is greyscaled, minimum threshold to call a cell
            returns:
            * markers: 2D array with integers marking pixels that correspond to individual cells
        '''   
        
        # Use blue channel of DAPI (BGR) to determine nuclei boundaries using HCS
        nuc_segmentations = self.pred_nuclei([img_nuclear[...,0].squeeze()])
        
        markers = label_nuclei(nuc_segmentations[0], img_full, 
            NUCLEUS_THRESHOLD = NUCLEUS_THRESHOLD,
            BORDER_THRESHOLD = BORDER_THRESHOLD,
            IMAGE_THRESHOLD = IMAGE_THRESHOLD,
            nuc_small_obj_size = nuc_small_obj_size, 
            cell_small_obj_size = cell_small_obj_size,
            cell_small_hole_size = cell_small_hole_size)
        
        return(markers)