# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:50 2024

@author: Joel Tapia Salvador
"""
import os
import re


from binarise_image import binarise
from cropping_license_plate import Cropper
from general_utils import read_image, show_image_on_window
from segment_characters import get_segmented_characters


PATH_TO_ORIGINAL_IMAGE_DIRECTORY = "../Lateral"
PATH_TO_RESULT_IMAGE_DIRECTORY = "../Results/Final"
WINDOW_SIZE = 1000
SHOW_RESULTS = False

FILE_REGEX_PATTERN = re.compile(r"(?:\.[a-zA-Z0-9]+$)")
ACCEPTED_IMAGE_FORMATS = (".png", ".jpg", ".jpge")

CROPPER_MODEL_PATH = "model.pt"


def main():
    """
    Exeucutes main bucle of the program to read license plates.

    Raises
    ------
    FileNotFoundError
        If there is a problem finding the images files.

    Returns
    -------
    None.

    """

    if not os.path.isdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY):
        raise FileNotFoundError(
            f'"{PATH_TO_ORIGINAL_IMAGE_DIRECTORY}" is not a directory.')

    cropper = Cropper(CROPPER_MODEL_PATH)

    for file_name in os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY):
        if FILE_REGEX_PATTERN.search(file_name).group() in ACCEPTED_IMAGE_FORMATS:
            original_image = read_image(
                PATH_TO_ORIGINAL_IMAGE_DIRECTORY, file_name)

            if SHOW_RESULTS:
                show_image_on_window(original_image, file_name, WINDOW_SIZE)

            cropped_license_plate = cropper.crop_image(original_image)

            if SHOW_RESULTS:
                show_image_on_window(cropped_license_plate,
                                     f"Cropped {file_name}", WINDOW_SIZE)

            binarised_license_plate = binarise(cropped_license_plate)

            if SHOW_RESULTS:
                show_image_on_window(binarised_license_plate,
                                     f"Binarised {file_name}", WINDOW_SIZE)

            segmented_characters = get_segmented_characters(
                binarised_license_plate)

            if SHOW_RESULTS:
                for i in range(len(segmented_characters)):
                    name_window = f"Character {
                        i + 1}/{len(segmented_characters)} {file_name}"
                    show_image_on_window(segmented_characters[i],
                                         name_window, WINDOW_SIZE)


if __name__ == "__main__":
    main()
