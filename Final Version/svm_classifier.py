# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:04:54 2024

@author: Joel Tapia Salvador
"""
import joblib
import warnings
import numpy as np

from typing import List

from image_transformations import binary_threshold, resize, to_gray

warnings.filterwarnings("ignore")


class SVM_Classifier():

    __slots__ = ("__model", "__threshold")

    def __init__(self, model_filename: str, threshold: int):
        self.__model = joblib.load(model_filename)
        self.__threshold = threshold

    def __adapt_characters(self, segmented_characters: List[np.array]):
        for i in range(len(segmented_characters)):
            segmented_characters[i] = self.__adapt_image(
                segmented_characters[i])
        segmented_characters = np.array(segmented_characters)
        segmented_characters = segmented_characters.reshape(
            segmented_characters.shape[0], -1)
        return segmented_characters

    def __adapt_image(self, image: np.array) -> np.array:
        if len(image.shape) == 3:
            image = to_gray(image)
        image_binary = binary_threshold(image, self.__threshold, 255)
        if image_binary.shape != (22, 36):
            image_binary = resize(image_binary, 22, 36)
        return image_binary/255

    def classify_characters(self, segmented_characters: List[np.array]) -> str:
        """
        Classify the characters from image into string using a SVM.

        Parameters
        ----------
        segmented_characters : List[numpy arrays]
            List of numpy array with the images of each segmented character to
            classify.

        Returns
        -------
        plate : string
            String representing the list of segmented characters. The order of
            the string characters is the order of the segmented characters.

        """
        segmented_characters = self.__adapt_characters(segmented_characters)
        predictions = self.__model.predict(segmented_characters)
        plate = ''.join(predictions)
        return plate


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
