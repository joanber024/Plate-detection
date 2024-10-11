# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:50 2024

@author: Joel Tapia Salvador
"""
import json
import os

from general_utils import remove_directory
from license_plate_reader import LicensePlateReader

import warnings

warnings.filterwarnings("ignore")

CROPPER_MODEL_PATH = os.path.join(".", "model.pt")
PATH_TO_ORIGINAL_IMAGE_DIRECTORY = os.path.join("..", "Lateral")
PATH_TO_RESULT_IMAGE_DIRECTORY = os.path.join(
    "..", "Results", "Final", "mat_españolas")
PATH_TO_MODEL = os.path.join('..', 'Classifier', 'svm_model.pkl')
WINDOW_SIZE = 1000

PROCESS_PAST_FAILED_ONLY = False
SAVE_RESULTS = True
SHOW_RESULTS = False


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

    if SAVE_RESULTS:
        if os.path.isdir(PATH_TO_RESULT_IMAGE_DIRECTORY):
            if len(os.listdir(PATH_TO_RESULT_IMAGE_DIRECTORY)):

                while True:
                    print('\33[31m' + f'"{PATH_TO_RESULT_IMAGE_DIRECTORY}" ' +
                          'has files already.\n Continue and ' +
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
                                              WINDOW_SIZE, 
                                              PATH_TO_MODEL)
    if PROCESS_PAST_FAILED_ONLY:
        with open("failed.dat", "r") as file:
            list_of_files = json.load(file)
    else:
        list_of_files = os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY)

    license_plates = reader_license_plate.read_license_plates(list_of_files,
                                                              PATH_TO_ORIGINAL_IMAGE_DIRECTORY)
    

    print(f'\n{test(list_of_files, license_plates):%}')


def test(list_of_files, license_plates):

    cum_punt = 0

    failed = []

    if not len(list_of_files) != len(license_plates):
        for file_name, license_plate in license_plates.items():
            if license_plate == 7:
                cum_punt += 1
            elif license_plate > 7:
                failed.append(file_name)

    score = cum_punt / len(list_of_files)

    if SAVE_RESULTS:
        with open("result.dat", "a") as file:
            file.write(f'{score}\n')

        with open("failed.dat", "w") as file:
            json.dump(failed, file)

    return score


if __name__ == "__main__":
    main()
