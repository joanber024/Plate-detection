# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:50 2024

@author: Joel Tapia Salvador
"""
import os

from general_utils import remove_directory
from license_plate_reader import LicensePlateReader

PATH_TO_ORIGINAL_IMAGE_DIRECTORY = os.path.join("..", "Lateral")
PATH_TO_RESULT_IMAGE_DIRECTORY = os.path.join("..", "Results", "Final")
WINDOW_SIZE = 1000

SAVE_RESULTS = True
SHOW_RESULTS = False

CROPPER_MODEL_PATH = os.path.join(".", "model.pt")


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
                print('\33[31m' + f'"{PATH_TO_RESULT_IMAGE_DIRECTORY}" has ' +
                      'files already.\n Continue and ' +
                      '\33[4m' + 'delete' + '\33[0m\33[31m' +
                      ' all files and directories in it?' + '\33[0m')
                option = input('Choose "Yes" or "No": ').lower()

                if option in ("no", "n", "-"):
                    print('\33[32m' + 'Exiting' + '\33[0m')
                    raise KeyboardInterrupt("User canceled")

                if option in ("yes", "y", "+"):
                    print('\33[32m' + 'Continuing' + '\33[0m')
                    remove_directory(PATH_TO_RESULT_IMAGE_DIRECTORY)

                    break

    reader_license_plate = LicensePlateReader(CROPPER_MODEL_PATH,
                                              PATH_TO_RESULT_IMAGE_DIRECTORY,
                                              SAVE_RESULTS,
                                              SHOW_RESULTS,
                                              WINDOW_SIZE)

    list_of_files = os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY)

    license_plates = reader_license_plate.read_license_plates(list_of_files,
                                                              PATH_TO_ORIGINAL_IMAGE_DIRECTORY)

    print(f'\n{test(list_of_files, license_plates) * 100}%')


def test(list_of_files, license_plates):

    score = 0

    if len(list_of_files) != len(license_plates):
        return 0

    for license_plate in license_plates:
        if license_plate == 7:
            score += 1

    return score / len(list_of_files)


if __name__ == "__main__":
    main()
