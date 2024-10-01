# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:50 2024

@author: Joel Tapia Salvador
"""
import os
import re


from binarise_image import binarise
from cropping_license_plate import Cropper
from general_utils import read_image, remove_directory, save_image, show_image_on_window
from segment_characters import get_segmented_characters


PATH_TO_ORIGINAL_IMAGE_DIRECTORY = os.path.join("..", "Lateral")
PATH_TO_RESULT_IMAGE_DIRECTORY = os.path.join("..", "Results", "Final")
WINDOW_SIZE = 1000

SAVE_RESULTS = True
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

    if os.path.isdir(PATH_TO_RESULT_IMAGE_DIRECTORY):
        if len(os.listdir(PATH_TO_RESULT_IMAGE_DIRECTORY)):

            while True:
                print('\33[31m' + f'"{PATH_TO_RESULT_IMAGE_DIRECTORY}" has files already.\n Continue and ' +
                      '\33[4m' + 'delete' + '\33[0m\33[31m' + ' all files and directories in it?' + '\33[0m')
                option = input('Choose "Yes" or "No": ').lower()

                if option in ("no", "n", "-"):
                    print('\33[32m' + 'Exiting' + '\33[0m')
                    raise KeyboardInterrupt("User canceled")

                if option in ("yes", "y", "+"):
                    print('\33[32m' + 'Continuing' + '\33[0m')
                    remove_directory(PATH_TO_RESULT_IMAGE_DIRECTORY)

                    break

    cropper = Cropper(CROPPER_MODEL_PATH)

    for file_name in os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY):

        file_format = FILE_REGEX_PATTERN.search(file_name).group()

        if file_format not in ACCEPTED_IMAGE_FORMATS:
            raise UserWarning(
                f'"{file_format}" is not an accepted image format. ')

        original_image = read_image(
            PATH_TO_ORIGINAL_IMAGE_DIRECTORY, file_name)

        if SHOW_RESULTS:
            show_image_on_window(original_image, file_name, WINDOW_SIZE)

        cropped_license_plate = cropper.crop_image(original_image)

        if SAVE_RESULTS:
            path = os.path.join(
                PATH_TO_RESULT_IMAGE_DIRECTORY, "Cropped License Plate")
            save_image(cropped_license_plate, path, file_name)
            del path

        if SHOW_RESULTS:
            show_image_on_window(cropped_license_plate,
                                 f"Cropped {file_name}", WINDOW_SIZE)

        binarised_license_plate = binarise(cropped_license_plate)

        if SAVE_RESULTS:
            path = os.path.join(
                PATH_TO_RESULT_IMAGE_DIRECTORY, "Binarised License Plate")
            save_image(binarised_license_plate, path, file_name)
            del path

        if SHOW_RESULTS:
            show_image_on_window(binarised_license_plate,
                                 f"Binarised {file_name}", WINDOW_SIZE)

        segmented_characters = get_segmented_characters(
            binarised_license_plate)

        if SAVE_RESULTS:
            for i in range(len(segmented_characters)):
                path = os.path.join(
                    PATH_TO_RESULT_IMAGE_DIRECTORY, "Segmented Characters",
                    file_name)
                save_image(segmented_characters[i], path, str(i) + file_format)
                del path

        if SHOW_RESULTS:
            for i in range(len(segmented_characters)):
                name_window = f"Character {
                    i + 1}/{len(segmented_characters)} {file_name}"
                show_image_on_window(segmented_characters[i],
                                     name_window, WINDOW_SIZE)


if __name__ == "__main__":
    main()
