# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:22:54 2024

@author: Joel Tapia Salvador
"""
import json

from typing import List, Dict

from main import main, SAVE_RESULTS, PATH_TO_ORIGINAL_IMAGE_DIRECTORY


def test(context: List) -> Dict:
    """
    Test the results of the main program.

    Parameters
    ----------
    context : List
        List with context of the execution, like files, results, etc.

    Returns
    -------
    scores : Dict
        Dictionary with the scores of the tests.

    """
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
        if not (cropped_license_plate == 0).all():
            cum_punt['cropped_license_plates'] += 1
        else:
            failed.add(file_name)

    for file_name, binarised_license_plate in results['binarised_license_plates'].items():
        if not (binarised_license_plate == 0).all():
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
        'absolute_score': {part: punt / len(list_files)
                           for part, punt in cum_punt.items()},
        'relative_score': {
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
