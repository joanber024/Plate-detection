# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:52:36 2024

@author: Joel Tapia Salvador
"""
import cv2

import numpy as np


from Typing import List


def __filter_contours(image: np.array):
    """
    Get the countours of the characted and filters them.

    Parameters
    ----------
    image : numpy array
        Image to detected the characters in represented as numpy array.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    contours, hierachy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    height, width = image.shape
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if (x == 0 or y == 0 or x+w == width or y+h == height):
            continue
        else:
            if (area > 700 and area < 30000):
                c = np.squeeze(c)
                xmax, xmin = np.max(c[:, 0]), np.min(c[:, 0])
                ymax, ymin = np.max(c[:, 1]), np.min(c[:, 1])
                prop = (ymax-ymin)/(xmax-xmin)
                if (prop > 1.5 and prop < 7):
                    filtered_contours.append(c)
    return filtered_contours[::-1]


def get_segmented_characters(image: np.array) -> List[np.array]:
    """
    Get the images of the segmented characters.

    Parameters
    ----------
    image : list
        Image to get the segmented characters from.

    Returns
    -------
    segmented_characters : List[numpy array]
        List of images represented as numpy array where each image is a
        character.

    """
    f_conts = __filter_contours(image)
    segmented_characters = []
    for c in f_conts:
        c = np.squeeze(c)
        xmax, xmin = np.max(c[:, 0]), np.min(c[:, 0])
        ymax, ymin = np.max(c[:, 1]), np.min(c[:, 1])
        segmented_characters.append(image[ymin:ymax, xmin:xmax])
    return segmented_characters
