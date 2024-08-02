import os
import logging
import sys
import time
import shutil
import pandas as pd
from pathlib import Path
import zipfile
import unittest
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


# Finds CSV's in a given path and adds them to blacklist
# Then sends blacklist to move_csvs
def get_list_of_csvs(regression):
    blacklist = []
    # for root, dirs, files in os.walk(extraction_path_datasets):
    for root, dirs, files in os.walk("D:/PycharmProjects/FinalYearProject/test/getListOfCsvs/"):
        for f in files:
            print(f"{f} is being evaluated")
            if f.endswith(".csv"):
                blacklist.append(os.path.join(root, f))
                logger.info(f"__get_list_of_csvs: Added file to blacklist --> {f}")
    if regression:
        move_csvs(blacklist)
    else:
        return blacklist


# Gets all csvs in a given path and adds them to a list
# Then sends that list to move_csvs, so they are moved to a new directory
def get_list_of_csvsx(regression):
    blacklist = []  # empty list to be filled with absolute path of all files that are csvs
    for file in os.listdir("D:/PycharmProjects/FinalYearProject/test/getListOfCsvs/"):  # for each file in given directory
        print(f"{file} is being evaluated")
        if file.endswith(".csv"):  # if file has extension .csv
            blacklist.append(f"D:/PycharmProjects/FinalYearProject/test/getListOfCsvs/{file}")  # add path to list
            logger.info(f"get_list_of_csvs: Added file to blacklist --> {file}")
    if regression:
        move_csvs(blacklist)  # send list to method that will move the csvs
    else:
        return blacklist


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


def delete_unused_files(path):
    logger.warning(f"This directory will be DELETED in path --> {path}, in 2 seconds...")
    time.sleep(2)
    shutil.rmtree(path)
    logger.info(f"Deletion complete in path --> {path}")


# Removes the top line from the CSV's
def reformat_csvs(path):
    # list_of_files = os.listdir(reformat_csvs_path)
    list_of_files = os.listdir(path)
    for file in list_of_files:
        logger.info(f"__reformat_csvs: dropping first row for file --> {file}")
        with open(f"{path}/{file}", 'r') as f:
            with open(f"{path}/updated_{file}", 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

    # new_list_of_files = os.listdir(reformat_csvs_path)
    new_list_of_files = os.listdir(path)
    for file in new_list_of_files:
        if not file.startswith("updated_"):
            logger.info(f"__reformat_csvs: removing file --> {file}")
            os.remove(f"{path}/{file}")
    remove_csvs_with_non_numerical_data(path)


def remove_csvs_with_non_numerical_data(path):
    # list_of_files = os.listdir(reformat_csvs_path)
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
        except:
            logger.info(f"remove_csvs_with_non_numerical_data: File--> {file} conversion failed, deleting")
            os.remove(f"{path}/{file}")


test_csv_reformatting = "D:/PycharmProjects/FinalYearProject/test/csvDatasets"
reformat_csvs(test_csv_reformatting)


class TestDataManager(unittest.TestCase):
    def test_get_csvs(self):
        data = get_list_of_csvs(False)
        result = len(data)
        self.assertEquals(result, 5, f"Method did not find all csvs...expected 5 but got --> {result}")

