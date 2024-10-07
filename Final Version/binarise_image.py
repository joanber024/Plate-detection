# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:48:01 2024

@author: Joel Tapia Salvador
"""
import numpy as np
import cv2


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
    result_image = fill_image(result_image)
    
    
    return result_image


def fill_image(original_image: np.array) -> np.array:
    for i, row in enumerate(original_image):
        values, freq = np.unique(row, return_counts=True)
        
        # Calcular el porcentaje de aparición de cada valor
        percs = (freq / len(row)) * 100
        for value, perc in zip(values, percs):
            if(value == 255 and perc > 80):
                original_image[i, :] = 0
    return original_image