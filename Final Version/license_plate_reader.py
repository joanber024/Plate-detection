# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:34:25 2024

@author: Joel Tapia Salvador
"""
import os
import re

from typing import List
from binarise_image import binarise
from cropping_license_plate import Cropper
from general_utils import create_path, read_image, save_image, show_image_on_window
from segment_characters import get_segmented_characters


class LicensePlateReader():

    __slots__ = ("__accepted_image_formats", "__cropper",
                 "__file_regex_pattern", "__path_to_result_image_directory",
                 "__save_results", "__show_results", "__window_size")

    def __init__(self,
                 cropper_model_path: str,
                 path_to_result_image_directory: str = ".",
                 save_results: bool = False,
                 show_results=False,
                 window_size: int = 1000):

        self.__accepted_image_formats = (".png", ".jpg", ".jpge")
        self.__file_regex_pattern = re.compile(r"(?:\.[a-zA-Z0-9]+$)")

        self.__cropper = Cropper(cropper_model_path)
        self.__path_to_result_image_directory = path_to_result_image_directory
        self.__save_results = save_results
        self.__show_results = show_results
        self.__window_size = window_size

    def read_license_plates(self,
                            list_of_files: List[str],
                            path_to_original_image_directory: str) -> List[str]:
        results = {}

        for file_name in list_of_files:

            file_format = self.__file_regex_pattern.search(file_name).group()

            if file_format not in self.__accepted_image_formats:
                raise UserWarning(
                    f'"{file_format}" is not an accepted image format. ')

            print(f'\nProcessing {file_name}:')

            original_image = read_image(
                path_to_original_image_directory, file_name)

            if self.__show_results:
                show_image_on_window(
                    original_image, file_name, self.__window_size)

            cropped_license_plate = self.__cropper.crop_image(original_image)

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

            binarised_license_plate = binarise(cropped_license_plate)

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

            segmented_characters = get_segmented_characters(
                binarised_license_plate)

            num_segmented_characters = len(segmented_characters)

            results[file_name] = num_segmented_characters

            print(f'\nNumber Segmented Characters: {num_segmented_characters}')

            if self.__save_results:
                path = os.path.join(
                    self.__path_to_result_image_directory,
                    "Segmented Characters",
                    file_name)

                create_path(path)

                for i in range(num_segmented_characters):
                    save_image(
                        segmented_characters[i],
                        path,
                        f'{i + 1} of {num_segmented_characters}' + file_format)

                del path

            if self.__show_results:
                for i in range(num_segmented_characters):
                    name_window = f"Character {i + 1}/{num_segmented_characters} {file_name}"
                    show_image_on_window(segmented_characters[i],
                                         name_window, self.__window_size)

        return results
