# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:48:01 2024

@author: Joel Tapia Salvador
"""
import numpy as np


from image_transformations import gaussian_blur, inverse, otsu_threshold, to_hsv


def binarise(original_image: np.array) -> np.array:
    """
    Binarise an image.

    Parameters
    ----------
    original_image : numpy array
        Image to be binarised represented as numpy array.

    Returns
    -------
    result_image : numpy array
        Binarised image represented as numpy array.

    """
    hsv_image = to_hsv(original_image)[:, :, 2]

    blured_image = gaussian_blur(hsv_image, (5, 5))

    threshold_image = otsu_threshold(blured_image, 0, 255)

    result_image = inverse(threshold_image)

    return result_image
