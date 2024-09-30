# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:33:56 2024

@author: Joel Tapia
"""
import numpy as np

from ultralytics import YOLO


class Cropper():
    __slots__ = ("__model")

    def __init__(self, model: str):
        self.__model = YOLO("best_model_1.pt")

    def crop_image(self, image: np.array) -> np.array:
        """
        Locates the license plate and crops the image.

        Parameters
        ----------
        image : numpy array
            Image represented as a numpy array.

        Returns
        -------
        cropped_image : numpy array
            Image with the cropped license plate represented as a numpy array.

        """
        result = self.__model(image)

        boxes = result.boxes
        bounding_box = np.array(boxes.xyxy, dtype='int32')[0]  # Coordinates
        cropped_image = image[bounding_box[1]:bounding_box[3],
                              bounding_box[0]:bounding_box[2]]

        return cropped_image
