# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:52:36 2024

@author: Joel Tapia Salvador
"""
import cv2

import numpy as np


from typing import List


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
    height, width = image.shape

    contours = []

    margin = 0
    crop_image = image

    while len(contours) < 7 and crop_image.size > 1000:

        crop_image = image[margin:height-margin, margin:width-margin]

        contours, hierachy = cv2.findContours(
            crop_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        margin += 1

    filtered_contours = []

    area_image = height * width

    # print(area_image)

    # print(len(contours))

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (x == 0 or y == 0 or x+w == width or y+h == height):
            area = cv2.contourArea(c)
            area_proportions = area / area_image
            if (area_proportions > 0.003 and area_proportions < 0.09):
                c = np.squeeze(c)
                xmax, xmin = np.max(c[:, 0]), np.min(c[:, 0])
                ymax, ymin = np.max(c[:, 1]), np.min(c[:, 1])
                prop = (ymax-ymin)/(xmax-xmin)
                if (prop > 1.5 and prop < 7):
                    filtered_contours.append(c)

    return crop_image, __order_contours(filtered_contours)


def __order_contours(contours: List) -> List:
    return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


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
    # print(image.shape)

    crop_image, f_conts = __filter_contours(image)

    # print(len(f_conts))

    segmented_characters = []
    for c in f_conts:
        c = np.squeeze(c)
        xmax, xmin = np.max(c[:, 0]), np.min(c[:, 0])
        ymax, ymin = np.max(c[:, 1]), np.min(c[:, 1])
        segmented_characters.append(crop_image[ymin:ymax, xmin:xmax])
    return segmented_characters
