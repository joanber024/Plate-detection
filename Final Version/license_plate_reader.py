# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:34:25 2024

@author: Joel Tapia Salvador
"""
import os
import re

import json
from typing import Dict, List
from binarise_image import binarise
from cropping_license_plate import Cropper
from general_utils import create_path, read_image, save_image, show_image_on_window
from segment_characters import get_segmented_characters
from svm_classifier import SVM_Classifier

separator = "\n" + "=" * os.get_terminal_size()[0]


class LicensePlateReader():

    __slots__ = ("__accepted_image_formats", "__classifier", "__cropper",
                 "__file_regex_pattern", "__path_to_result_image_directory",
                 "__save_results", "__show_results", "__window_size",
                 "__model_filename")

    def __init__(self,
                 cropper_model_path: str,
                 classifier_model_path: str,
                 path_to_result_image_directory: str = ".",
                 save_results: bool = False,
                 show_results: bool = False,
                 threshold_classifier: int = 127,
                 window_size: int = 1000):

        self.__accepted_image_formats = (".png", ".jpg", ".jpge")
        self.__file_regex_pattern = re.compile(r"(?:\.[a-zA-Z0-9]+$)")

        self.__cropper = Cropper(cropper_model_path)
        self.__classifier = SVM_Classifier(
            classifier_model_path, threshold_classifier)
        self.__path_to_result_image_directory = path_to_result_image_directory
        self.__save_results = save_results
        self.__show_results = show_results
        self.__window_size = window_size

    def read_license_plates(self,
                            list_of_files: List[str],
                            path_to_original_image_directory: str) -> Dict:
        results = {
            'original_images': {},
            'cropped_license_plates': {},
            'binarised_license_plates': {},
            'segmented_characters': {},
            'license_plates': {}
        }

        for file_name in list_of_files:

            # Process file format

            file_format = self.__file_regex_pattern.search(file_name).group()

            if file_format not in self.__accepted_image_formats:
                raise UserWarning(
                    f'"{file_format}" is not an accepted image format. ')

            print(separator)

            print('\nProcessing ' + '\33[4m' + f'{file_name}:' + '\33[0m')

            # Read image from storage

            original_image = read_image(
                path_to_original_image_directory, file_name)

            results['original_images'][file_name] = original_image

            if self.__show_results:
                show_image_on_window(
                    original_image, file_name, self.__window_size)

            # Crop license plate

            cropped_license_plate = self.__cropper.crop_image(original_image)

            results['cropped_license_plates'][file_name] = cropped_license_plate

            if self.__save_results:
                path = os.path.join(
                    self.__path_to_result_image_directory,
                    "Cropped License Plate")

                create_path(path)

                save_image(cropped_license_plate, path, file_name)

                del path

            if self.__show_results:
                show_image_on_window(cropped_license_plate,
                                     f"Cropped {file_name}",
                                     self.__window_size)

            # Binarise license plate

            binarised_license_plate = binarise(cropped_license_plate)

            results['binarised_license_plates'][file_name] = binarised_license_plate

            if self.__save_results:
                path = os.path.join(
                    self.__path_to_result_image_directory,
                    "Binarised License Plate")

                create_path(path)

                save_image(binarised_license_plate, path, file_name)

                del path

            if self.__show_results:
                show_image_on_window(binarised_license_plate,
                                     f"Binarised {file_name}",
                                     self.__window_size)

            # Segment characters

            segmented_characters = get_segmented_characters(
                binarised_license_plate)

            results['segmented_characters'][file_name] = segmented_characters

            num_segmented_characters = len(segmented_characters)

            print(f'\nNumber Segmented Characters: {num_segmented_characters}')

            if self.__show_results:
                for i in range(num_segmented_characters):
                    name_window = f"Character {
                        i + 1}/{num_segmented_characters} {file_name}"
                    show_image_on_window(segmented_characters[i],
                                         name_window, self.__window_size)

            if self.__save_results:

                title = ''

                if num_segmented_characters == 7:
                    classification = "Exactly 7"

                elif num_segmented_characters < 7:
                    classification = "Less 7"

                elif num_segmented_characters > 7:
                    classification = "More 7"

                path = os.path.join(
                    self.__path_to_result_image_directory,
                    "Segmented Characters", classification,
                    file_name)

                create_path(path)

                for i in range(num_segmented_characters):
                    if num_segmented_characters == 7:
                        title = f' [{file_name[i]}]'
                    save_image(
                        segmented_characters[i],
                        path, f'{i + 1} of {num_segmented_characters}' +
                        title + file_format)

                del path, classification, title

            # Classify characters

            if num_segmented_characters == 7:
                plate = self.__classifier.classify_characters(
                    segmented_characters)

                print('\n' + '\33[32m' +
                      'PLATE RECOGNISED:' + '\33[0m' + plate)
            else:
                plate = ''

                print('\n' + '\33[31m' + 'PLATE NOT RECOGNISED' + '\33[0m')

            results['license_plates'][file_name] = plate

            if self.__save_results:
                path = os.path.join(
                    self.__path_to_result_image_directory,
                    'license_plates.log')
                with open(path, 'a') as file:
                    file.write(f'"{file_name}": {plate}\n')

                del path

        print(separator)

        return results


if __name__ == "__main__":
    print(
        '\33[31m' + 'You are executing a module file, execute main instead.'
        + '\33[0m')
