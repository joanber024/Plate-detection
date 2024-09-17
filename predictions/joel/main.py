# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:25:29 2024

@author: JoelT
"""

import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from typing import Tuple

PATH_TO_ORIGINAL_IMAGE_DIRECTORY = "./Lateral"
PATH_TO_RESULT_IMAGE_DIRECTORY = "./Processed"
WINDOW_SIZE = 1000
SHOW_RESULTS = False


def to_gray(image: np.array) -> np.array:
    # return 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blackhat(image: np.array, kernel_size: Tuple[int, int]) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def inverse(image: np.array) -> np.array:
    return 255 - image


def median_blur(image: np.array, kernel_size: int) -> np.array:
    return cv2.medianBlur(image, kernel_size)


def gaussian_blur(image: np.array, kernel_size: Tuple[int, int], mu:int = 0):
    return cv2.GaussianBlur(image, kernel_size, mu)
    

def dilate(image: np.array, kernel_size: Tuple[int, int], iterations: int) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.dilate(image, kernel, iterations)


def gradient(image: np.array, kernel_size: Tuple[int, int]) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def opening(image: np.array, kernel_size: Tuple[int, int]) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image: np.array, kernel_size: Tuple[int, int]) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def threshold(image: np.array, threshold: int, max_value:int) -> np.array:
    _, result = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY) 
    
    return result


def otsu_threshold(image: np.array, threshold: int, max_value:int) -> np.array:
    _, result = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return result


def inv_threshold(image: np.array, threshold: int, max_value:int) -> np.array:
    _, result = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV) 
    
    return result


def tophat(image: np.array, kernel_size: Tuple[int, int]) -> np.array:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)



def show_image_on_window(image: np.array, window_name: str = "Image") -> None:
    scale = max(image.shape[:2]) / WINDOW_SIZE
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(
        image.shape[1] // scale), int(image.shape[0] // scale))
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    
    
def process_image(original_image: np.array) -> np.array:
    # gray_image = to_gray(original_image)
    
    # dialted_image_1 = dilate(gray_image, (3, 3), 1)
    
    # opening_image_1 = opening(dialted_image_1, (10, 10))
    
    # dialted_image_2 = dilate(opening_image_1, (5, 5), 1)
    
    # blackhat_image = blackhat(dialted_image_2, (45, 30))
    
    # closing_image_1 = closing(blackhat_image, (5, 5))
    
    # blured_image_1 = median_blur(closing_image_1, 9)
    
    # threshold_image_2 = threshold(blured_image_1, 120, 255)
    
    # closing_image_2 = closing(threshold_image_2, (17, 17))
    
    # dialted_image_3 = dilate(closing_image_2, (5, 5), 1)
    
    # return dialted_image_3
    
    gray_image = to_gray(original_image)
    
    blured_image = gaussian_blur(gray_image, (5, 5))
    
    threshold_image = otsu_threshold(blured_image, 0, 255)
    
    inverted_image = inverse(threshold_image)
    
    closing_image_1 = closing(inverted_image, (10, 10))
    
    return closing_image_1
    

if __name__ == "__main__":
    if os.path.isdir(PATH_TO_RESULT_IMAGE_DIRECTORY):
        if len(os.listdir(PATH_TO_RESULT_IMAGE_DIRECTORY)) > 0:
            for file_name in os.listdir(PATH_TO_RESULT_IMAGE_DIRECTORY):
                os.remove(PATH_TO_RESULT_IMAGE_DIRECTORY + "/" + file_name)
            del(file_name)  
    else:
        os.mkdir(PATH_TO_RESULT_IMAGE_DIRECTORY)
    
    for file_name in os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY):
        original_image = cv2.imread(PATH_TO_ORIGINAL_IMAGE_DIRECTORY + "/" + file_name)
    
        result_image = process_image(original_image)
    
        cv2.imwrite(PATH_TO_RESULT_IMAGE_DIRECTORY+ "/" + file_name, result_image)
    
        if SHOW_RESULTS:    
            show_image_on_window(result_image)
