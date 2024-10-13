# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:32:50 2024

@author: Joel Tapia Salvador
"""
import json
import os

from general_utils import remove_directory
from license_plate_reader import LicensePlateReader

CLASSIFIER_MODEL_PATH = os.path.join('.', 'svm_model.pkl')
CROPPER_MODEL_PATH = os.path.join(".", "model.pt")
PATH_TO_ORIGINAL_IMAGE_DIRECTORY = os.path.join("..", "mat_españolas")
PATH_TO_RESULT_IMAGE_DIRECTORY = os.path.join(
    "..", "Results", "Final", "mat_españolas")
THRESHOLD = 127
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
                    print('\n' + '\33[31m' +
                          f'"{PATH_TO_RESULT_IMAGE_DIRECTORY}" ' +
                          'has files already.\n Continue and ' +
                          '\33[4m' + 'delete' + '\33[0m\33[31m' +
                          ' all files and directories in it?' + '\33[0m')
                    option = input('\nChoose "Yes" or "No": ').lower()

                    if option in ("no", "n", "-"):
                        print('\n' + '\33[31m' + 'Exiting' + '\33[0m')
                        raise KeyboardInterrupt("User canceled")

                    if option in ("yes", "y", "+"):
                        print('\n' + '\33[32m' + 'Continuing' + '\33[0m')
                        remove_directory(PATH_TO_RESULT_IMAGE_DIRECTORY)

                        break

    reader_license_plate = LicensePlateReader(CROPPER_MODEL_PATH,
                                              CLASSIFIER_MODEL_PATH,
                                              PATH_TO_RESULT_IMAGE_DIRECTORY,
                                              SAVE_RESULTS,
                                              SHOW_RESULTS,
                                              THRESHOLD,
                                              WINDOW_SIZE)
    if PROCESS_PAST_FAILED_ONLY:
        with open("failed.dat", "r") as file:
            list_of_files = json.load(file)
    else:
        list_of_files = os.listdir(PATH_TO_ORIGINAL_IMAGE_DIRECTORY)

    results = reader_license_plate.read_license_plates(list_of_files,
                                                       PATH_TO_ORIGINAL_IMAGE_DIRECTORY)

    return list_of_files, results


def test(context):

    list_files = context[0]

    results = context[1]

    cum_punt = {
        'cropped_license_plates': 0,
        'binarised_license_plates': 0,
        'segmented_characters': 0,
        'license_plates': 0
    }

    failed = set()

    for file_name, cropped_license_plate in results['cropped_license_plates'].items():
        if (cropped_license_plate != 0).all:
            cum_punt['cropped_license_plates'] += 1
        else:
            failed.add(file_name)

    for file_name, binarised_license_plate in results['binarised_license_plates'].items():
        if (binarised_license_plate != 0).all:
            cum_punt['binarised_license_plates'] += 1
        else:
            failed.add(file_name)

    for file_name, segmented_characters in results['segmented_characters'].items():
        if len(segmented_characters) == 7:
            cum_punt['segmented_characters'] += 1
        else:
            failed.add(file_name)

    for file_name, license_plate in results['license_plates'].items():
        if license_plate == file_name[:7]:
            cum_punt['license_plates'] += 1
        else:
            failed.add(file_name)

    failed = list(failed)

    scores = {
        'abs_score': {part: punt / len(list_files)
                      for part, punt in cum_punt.items()},
        'rel_score': {
            'cropped_license_plates': cum_punt['cropped_license_plates'] / len(list_files),
            'binarised_license_plates': cum_punt['binarised_license_plates'] / cum_punt['cropped_license_plates'],
            'segmented_characters': cum_punt['segmented_characters'] / cum_punt['binarised_license_plates'],
            'license_plates': cum_punt['license_plates'] / cum_punt['segmented_characters']
        }
    }

    if SAVE_RESULTS:
        with open("result.dat", "a") as file:
            file.write(f'{PATH_TO_ORIGINAL_IMAGE_DIRECTORY}: {scores}\n')

        with open("failed.dat", "w") as file:
            json.dump(failed, file)

    for typ, punts in scores.items():
        print(f'{typ}\n')
        for score, punt in punts.items():
            print(f'{score}: {punt: %}\n')

    return scores


if __name__ == "__main__":
    test(main())
