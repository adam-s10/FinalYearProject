import os
import logging
import sys
import time
import shutil
import pandas as pd
from pathlib import Path
import zipfile
import numpy as np

# Instantiate Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# new_path_for_csvs = "/Users/adam/PycharmProjects/FinalYearProject/csvDatasets"
new_path_for_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"
# download_path_zips = "/Users/adam/PycharmProjects/FinalYearProject/zips/"
download_path_zips = "D:/PycharmProjects/FinalYearProject/zips/"
# extraction_path_datasets = "/Users/adam/PycharmProjects/FinalYearProject/extractedZips/"
extraction_path_datasets = "D:/PycharmProjects/FinalYearProject/extractedZips/"
# reformat_csvs_path = "/Users/adam/PycharmProjects/FinalYearProject/csvDatasets"
reformat_csvs_path = "D:/PycharmProjects/FinalYearProject/csvDatasets"


def csv_cleanup_service():
    logger.info("csv_cleanup_service: Beginning Cleanup...")
    get_list_of_csvs()


# Finds CSV's in a given path and adds them to blacklist
# Then sends blacklist to move_csvs
def get_list_of_csvs():
    blacklist = []
    # for root, dirs, files in os.walk(extraction_path_datasets):
    for root, dirs, files in os.walk("D:/PycharmProjects/FinalYearProject/test/extractedZips/"):
        for f in files:
            print(f"{f} is being evaluated")
            if f.endswith(".csv"):
                blacklist.append(os.path.join(root, f))
                logger.info(f"__get_list_of_csvs: Added file to blacklist --> {f}")
    move_csvs(blacklist)


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
    # __delete_unused_files(extraction_path_datasets)
    delete_unused_files("D:/PycharmProjects/FinalYearProject/test/extractedZips/")
#     TODO: add remove file from zip directory


def delete_unused_files(path):
    try:
        logger.warning(f"This directory will be DELETED in path --> {path}, in 2 seconds...")
        time.sleep(2)
        shutil.rmtree(path)
        logger.info(f"Deletion complete in path --> {path}")
    except:
        logger.info(f"Deletion path--> {path} failed")


# Removes the top line from the CSV's
def reformat_csvs(path):
    # list_of_files = os.listdir(reformat_csvs_path)
    list_of_files = os.listdir(path)
    for file in list_of_files:
        logger.info(f"__reformat_csvs: dropping first row for file --> {file}")
        try:
            with open(f"{path}/{file}", 'r') as f:
                with open(f"{path}/updated_{file}", 'w') as f1:
                    next(f)  # skip header line
                    for line in f:
                        f1.write(line)
        except:
            logger.info(f"reformat_csvs: dropping first row failed for file --> {file}")

    # new_list_of_files = os.listdir(reformat_csvs_path)
    new_list_of_files = os.listdir(path)
    for file in new_list_of_files:
        if not file.startswith("updated_"):
            logger.info(f"__reformat_csvs: removing file --> {file}")
            os.remove(f"{path}/{file}")
    remove_csvs_with_non_numerical_data(path)


def remove_csvs_with_non_numerical_data(path):
    # list_of_files = os.listdir(reformat_csvs_path)
    removed_files = 0
    kept_files = 0
    list_of_files = os.listdir(path)
    for file in list_of_files:
        logger.info(f"remove_csvs_with_non_numerical_data: Reading file--> {file}")
        try:
            df = pd.read_csv(f"{path}/{file}", header=None)
            # print(df)
            array_2d = df.to_numpy()
            # print(array_2d)
            array_2d.astype(float)
            logger.info(f"remove_csvs_with_non_numerical_data: File--> {file} only contains numbers, will not be deleted")
            kept_files += 1
        except:
            logger.info(f"remove_csvs_with_non_numerical_data: File--> {file} conversion failed, deleting")
            os.remove(f"{path}/{file}")
            removed_files += 1

    logger.info(f"Number of kept files --> {kept_files}")
    logger.info(f"Number of removed files --> {removed_files}")


def extract_data(path_given):
    extracted_locations = []
    zip_location = os.walk("D:/PycharmProjects/FinalYearProject/zips")
    # for word in zip_location:
    #     # print(f"Here is the file name: {word}")
    #     # # file_name = word.split('\\')[hello world]
    #     # zipfile_location = f"{download_path_zips}{word}"
    #     extracted_to = f"{extraction_path_datasets}{word}"
    #     path = Path(zipfile_location)
    #     for i in path.glob("*.zip"):
    #         with zipfile.ZipFile(i, "r") as Zip:
    #             Zip.extractall(extracted_to)
    #
    #     extracted_locations.append(word)
    # directories = os.listdir(download_path_zips)
    extraction_failures = 0
    directories = os.listdir(path_given)
    for d in directories:
        print(d)
        # path = Path(f"{download_path_zips}{d}")
        path = Path(f"{path_given}/{d}")
        for i in path.glob("*.zip"):
            logger.info(f"File to be extracted --> {i}")
            # extracted_to = f"{extraction_path_datasets}{d}"
            extracted_to = f"D:/PycharmProjects/FinalYearProject/test/extractedZips/{d}"
            try:
                with zipfile.ZipFile(i, 'r') as Zip:
                    Zip.extractall(extracted_to)
                logger.info(f"File successfully extracted --> {i}")
            except:
                logger.info(f"Failed to Extract data--> {d}")
                extraction_failures += 1

        get_list_of_csvs()
        delete_unused_files(f"{path_given}/{d}")
    # print(extracted_locations)
    # reformat_csvs(path_given)
    logger.info(f"Extraction Failures --> {extraction_failures}")
    reformat_csvs("D:/PycharmProjects/FinalYearProject/test/csvDatasets")


# extract_data()
test_csv_reformatting = "D:/PycharmProjects/FinalYearProject/test/csvDatasets"
test_csv_reformatting_live_data = "C:/Users/Adam/FinalYearProject_TestData"
# reformat_csvs(test_csv_reformatting)
extract_data(test_csv_reformatting_live_data)
