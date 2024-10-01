# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:33:56 2024

@author: Joel Tapia
"""
import numpy as np

from typing import List
from ultralytics import YOLO


class Cropper():
    __slots__ = ("__model")

    def __init__(self, model: str):
        self.__model = YOLO(model)

    def __choose_best_result(self, results: List):
        if len(results) == 1:
            return results[0]
        else:
            raise NotImplementedError("Not implemented yet.")

    def __get_results(self, image: np.array) -> List:
        return self.__model(image)

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
        results = self.__get_results(image)

        result = self.__choose_best_result(results)

        boxes = result.boxes
        bounding_box = np.array(boxes.xyxy, dtype='int32')[0]  # Coordinates
        cropped_image = image[bounding_box[1]:bounding_box[3],
                              bounding_box[0]:bounding_box[2]]

        return cropped_image
