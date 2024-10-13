# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:33:56 2024

@author: Joel Tapia
"""
import os
import numpy as np

from typing import List
from ultralytics import YOLO


class Cropper():

    __slots__ = ("__name_model", "__model")

    def __init__(self, path_model: str):
        self.__name_model = os.path.split(path_model)[1]
        self.__model = YOLO(path_model)

    def __choose_best_result(self, bounding_boxes: np.array, image: np.array):
        if bounding_boxes.shape[0] == 1:
            return bounding_boxes[0]
        else:
            height, width, _ = image.shape

            image_center = np.array([width / 2, height / 2])

            result = None

            minimum = float("inf")

            for bounding_box in bounding_boxes:

                bounding_box_center = np.array([
                    (bounding_box[2] + bounding_box[0]) / 2,
                    (bounding_box[3] + bounding_box[1]) / 2])

                distance = np.sqrt(
                    np.sum(np.square(image_center - bounding_box_center)))

                if distance < minimum:
                    result = bounding_box
                    minimum = distance

            return result

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
        result = self.__get_results(image)[0]

        boxes = result.boxes

        bounding_boxes = np.array(boxes.xyxy, dtype='int32')

        bounding_box = self.__choose_best_result(bounding_boxes, image)

        if bounding_box is None:
            cropped_image = image.copy()
            cropped_image[:, :, :] = 0
            return cropped_image

        # Coordinates
        cropped_image = image[bounding_box[1]:bounding_box[3],
                              bounding_box[0]:bounding_box[2]]

        return cropped_image

    @property
    def model(self) -> str:
        """
        Property returning the name of the file the model was loaded from.

        Returns
        -------
        string
            Name of the file where the model was loaded from.

        """
        return self.__name_model


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
