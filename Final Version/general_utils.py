# -*- coding: utf-8 -*- noqa
"""
Created on Fri Sep 27 12:50:37 2024

@author: JoelT
"""

import os
import cv2
import numpy as np


def read_image(path: str, name: str) -> np.array:
    """
    Read the image from a file using OpenCV.

    Parameters
    ----------
    path : str
        Path to the image file.
    name : str
        Name of the image file.

    Returns
    -------
    numpy.array
        Numpy array representing the image.

    """
    return cv2.imread(os.ath.join(path, name))


def save_image(image: np.array, path: str, name: str) -> bool:
    """
    Save as a file a given image using OpenCV.

    Parameters
    ----------
    image : numpy.array
        Numpy array representing the image.
    path : str
        Path to the image file.
    name : str
        Nome of the image file.

    Returns
    -------
    bool
        If the funtion was able to save the image or not.

    """
    try:
        if not os.path.isdir(path):
            os.mkdir(path)

        cv2.imwrite(os.path.join(path, name), image)

        return True

    except:  # pylint: disable=bare-except
        return False


def show_image_on_window(image: np.array, window_name: str = "Image",
                         window_size: int = 1000) -> None:
    """
    Display the image on window.

    Parameters
    ----------
    image : numpy.array
        Numpy array representing the image.
    window_name : str, optional
        Name the window will have. The default is "Image".
    window_size : int, optional
        Max size of the window in pixels. The default is 1000 pixels.

    Returns
    -------
    None
        Returns none once the window is closed.

    """
    scale = max(image.shape[:2]) / window_size
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(
        image.shape[1] // scale), int(image.shape[0] // scale))
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
