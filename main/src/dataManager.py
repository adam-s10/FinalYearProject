import os
import logging
import sys
import time
import shutil

import deprecation
import pandas as pd
from pathlib import Path
import zipfile

# Instantiate Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

new_path_for_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"
download_path_zips = "D:/PycharmProjects/FinalYearProject/zips/"
extraction_path_datasets = "D:/PycharmProjects/FinalYearProject/extractedZips/"
reformat_csvs_path = "D:/PycharmProjects/FinalYearProject/csvDatasets"


# Gets all csvs in a given path and adds them to a list
# Then sends that list to move_csvs, so they are moved to a new directory
def get_list_of_csvs(extracted_path):
    blacklist = []  # empty list to be filled with absolute path of all files that are csvs
    for root, dirs, files in os.walk(extracted_path):  # for everything in given directory
        for f in files:  # for file in all files found by os.walk
            print(f"{f} is being evaluated")
            if f.endswith(".csv"):  # if file has extension .csv
                blacklist.append(os.path.join(root, f))  # add path to list
                logger.info(f"__get_list_of_csvs: Added file to blacklist --> {f}")
    move_csvs(blacklist)  # send list to method that will move the csvs


# Gets all csvs in a given path and adds them to a list
# Then sends that list to move_csvs, so they are moved to a new directory
@deprecation.deprecated(deprecated_in="Sprint 2",
                        details="Does not work as expected, cannot find files in children of provided directory."
                                "Use get_list_of_csvs in dataManager.py instead")
def get_list_of_csvs_old(extracted_path):
    blacklist = []  # empty list to be filled with absolute path of all files that are csvs
    for file in os.listdir(extracted_path):  # for each file in given directory
        print(f"{file} is being evaluated")
        if file.endswith(".csv"):  # if file has extension .csv
            blacklist.append(f"{extracted_path}/{file}")  # add path to list
            logger.info(f"get_list_of_csvs: Added file to blacklist --> {file}")
    move_csvs(blacklist)  # send list to method that will move the csvs


# Takes blacklist as the input containing a list of paths to csv files
# Moves them to a new directory to save them from deletion
def move_csvs(blacklist):
    for path in blacklist:
        temp = path.split("/")
        print(f"{new_path_for_csvs}/{len(temp) - 1}")
        # shutil.copy(path, new_path_for_csvs)
        logger.info(f"Moving file in blacklist --> {path} to --> {new_path_for_csvs}")
        shutil.copy(path, "D:/PycharmProjects/FinalYearProject/test/csvDatasets")
        print(type(path), path)
        logger.info(f"__move_csvs: Moved file {path} in blacklist to provided directory {new_path_for_csvs}")
    delete_unused_files(extraction_path_datasets)


def delete_unused_files(path):
    try:
        logger.warning(f"This directory will be DELETED in path --> {path}, in 2 seconds...")
        time.sleep(2)
        shutil.rmtree(path)
        logger.info(f"Deletion complete in path --> {path}")
    except:
        logger.info(f"Deletion path--> {path} failed")


# Removes the top line from the csvs saving them as a new file. Then removes the original csvs
def reformat_csvs(path):
    list_of_files = os.listdir(path)  # get all the csvs in the given path
    removed_files = 0  # count of failures
    for file in list_of_files:  # for each file in directory
        logger.info(f"reformat_csvs: dropping first row for file --> {file}")
        try:
            with open(f"{path}/{file}", 'r') as f:  # open the file in read mode
                with open(f"{path}/updated_{file}", 'w') as f1:  # open new file in format updated_ file name
                    next(f)  # skip header line
                    for line in f:  # for each line in original file
                        f1.write(line)  # write rest of the lines to new file
        except:
            logger.info(f"reformat_csvs: dropping first row failed for file --> {file}")
            os.remove(f"{path}/{file}")  # delete any file that fails
            removed_files += 1  # keep count of failed files

    logger.info(f"reformat_csvs: amount of removed files --> {removed_files}")
    new_list_of_files = os.listdir(path)  # get all files in the path, including the new ones
    for file in new_list_of_files:  # for each file in directory
        if not file.startswith("updated_"):  # if the file is not one with header removed
            logger.info(f"__reformat_csvs: removing file --> {file}")
            os.remove(f"{path}/{file}")  # remove file
    remove_csvs_with_non_numerical_data(path)  # start process to remove any datasets that are not purely numerical


# Attempts to convert csv to a float. If this fails file will be deleted
def remove_csvs_with_non_numerical_data(path):
    removed_files = 0
    kept_files = 0
    list_of_files = os.listdir(path)
    for file in list_of_files:
        logger.info(f"remove_csvs_with_non_numerical_data: Reading file--> {file}")
        try:
            df = pd.read_csv(f"{path}/{file}", header=None)
            # Conversion to float
            array_2d = df.to_numpy()
            array_2d.astype(float)
            logger.info(f"remove_csvs_with_non_numerical_data: File--> {file} only contains numbers, will not be deleted")
            kept_files += 1
        except:
            logger.info(f"remove_csvs_with_non_numerical_data: File--> {file} conversion failed, deleting")
            os.remove(f"{path}/{file}")
            removed_files += 1

    logger.info(f"Non Numerical Data: Number of kept files --> {kept_files}")
    logger.info(f"Non Numerical Data: Number of removed files --> {removed_files}")


# Extracts zip files in the provided directory one by one
# After a file is extracted, the csvs are identified using get_list_of_csvs
# They are moved and any remaining files are deleted
def extract_data(path_given):
    extraction_failures = 0  # count of failures
    directories = os.listdir(path_given)
    for d in directories:
        print(d)
        path = Path(f"{path_given}/{d}")  # path of child directory where zip resides
        for i in path.glob("*.zip"):  # for each file that is a zip
            logger.info(f"File to be extracted --> {i}")
            extracted_to = f"{extraction_path_datasets}{d}"  # declare location for data to be extracted to
            try:
                with zipfile.ZipFile(i, 'r') as Zip:  # unzip
                    Zip.extractall(extracted_to)  # extract data to new location
                logger.info(f"File successfully extracted --> {i}")
            except:
                logger.info(f"Failed to Extract data--> {d}")  # log and keep track of failures
                extraction_failures += 1

        # identifies and moves csvs, then removes the directory data was extracted to as it contains 0 csvs
        get_list_of_csvs(extraction_path_datasets)
        delete_unused_files(f"{path_given}/{d}")  # remove the child directory
    logger.info(f"Extraction Failures --> {extraction_failures}")
    reformat_csvs("D:/PycharmProjects/FinalYearProject/test/csvDatasets")


# extract_data()
# reformat_csvs(test_csv_reformatting)
# extract_data(test_csv_reformatting_live_data)
extract_data(download_path_zips)

