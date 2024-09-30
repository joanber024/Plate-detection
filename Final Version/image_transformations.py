# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:44:27 2024

@author: Joel Tapia Salvador
"""

import cv2
import numpy as np

from typing import Tuple


def gaussian_blur(image: np.array, kernel_size: Tuple[int, int],
                  mu: int = 0) -> np.array:
    """
    Blur an image applying Gaussian  Blur.

    Parameters
    ----------
    image : numpy array
        Image to be blured represented as numpy array.
    kernel_size : Tuple[interger, interger]
        Tuple of 2 integer numebers with the size of the kernel.
    mu : integer, optional
        Standard deviation of the Gaussian curve applied. The default is 0.

    Returns
    -------
    numpy array
        Blurred image represented as numpy array.

    """
    return cv2.GaussianBlur(image, kernel_size, mu)


def inverse(image: np.array) -> np.array:
    """
    Invert the colors of an image.

    Parameters
    ----------
    image : numpy array
        Image to be inverted represented as numpy array.

    Returns
    -------
    numpy array
        Inverse image represented as numpy array.

    """
    return 255 - image


def otsu_threshold(image: np.array, threshold: int, max_value: int) -> np.array:
    """
    Apply an OTSU threshold to an image.

    Parameters
    ----------
    image : numpy array
        Image to get thresholded represented as numpy array.
    threshold : integer
        Number that the threshold is applied over.
    max_value : integer
        Value the threshold changes to.

    Returns
    -------
    result : numpy array
        Thresholded image represented as numpy array.

    """
    _, result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return result


def to_hsv(image: np.array) -> np.array:
    """
    Change the colour space of an image from RGB to HSV.

    Parameters
    ----------
    image : numpy array
        Image to get converted to HSV colour space represented as numpy array.

    Returns
    -------
    numpy array
        Image in HSV colour space represented as numpy array.

    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
