import os
import logging
import sys
import time
import shutil
import pandas as pd
import numpy as np

# Instantiate Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

new_path_for_csvs = "/Users/adam/PycharmProjects/FinalYearProject/csvDatasets"
download_path_zips = "/Users/adam/PycharmProjects/FinalYearProject/zips/"
extraction_path_datasets = "/Users/adam/PycharmProjects/FinalYearProject/extractedZips/"
reformat_csvs_path = "/Users/adam/PycharmProjects/FinalYearProject/csvDatasets"


def csv_cleanup_service():
    logger.info("csv_cleanup_service: Beginning Cleanup...")
    __get_list_of_csvs()


def __get_list_of_csvs():
    blacklist = []
    for root, dirs, files in os.walk(extraction_path_datasets):
        for f in files:
            if f.endswith(".csv"):
                blacklist.append(os.path.join(root, f))
                logger.info(f"__get_list_of_csvs: Added file to blacklist --> {f}")
    __move_csvs(blacklist)


def __move_csvs(blacklist):
    for path in blacklist:
        temp = path.split("/")
        print(f"{new_path_for_csvs}/{len(temp) - 1}")
        shutil.copy(path, new_path_for_csvs)
        logger.info(f"__move_csvs: Moved file {path} in blacklist to provided directory {new_path_for_csvs}")
    __delete_unused_files(extraction_path_datasets)


def __delete_unused_files(path):
    logger.warning(f"This directory will be DELETED in path --> {path}, in 10 seconds...")
    time.sleep(10)
    shutil.rmtree(path)
    logger.info(f"Deletion complete in path --> {path}")


def __reformat_csvs():
    list_of_files = os.listdir(reformat_csvs_path)
    for file in list_of_files:
        logger.info(f"__reformat_csvs: dropping first row for file --> {file}")
        with open(f"{reformat_csvs_path}/{file}", 'r') as f:
            with open(f"{reformat_csvs_path}/updated_{file}", 'w') as f1:
                next(f)  # skip header line
                for line in f:
                    f1.write(line)

    new_list_of_files = os.listdir(reformat_csvs_path)
    for file in new_list_of_files:
        if not file.startswith("updated_"):
            logger.info(f"__reformat_csvs: removing file --> {file}")
            os.remove(f"{reformat_csvs_path}/{file}")
    __remove_csvs_with_non_numerical_data()


def __remove_csvs_with_non_numerical_data():
    list_of_files = os.listdir(reformat_csvs_path)
    for file in list_of_files:
        logger.info(f"Reading file: {file}")
        try:
            df = pd.read_csv(f"{reformat_csvs_path}/{file}")
            print(df)
            array_2d = df.to_numpy()
            print(array_2d)
            array_2d.astype(float)
        except:
            logger.info(f"File: {file} could not be read/converted to 2D array, deleting")
            os.remove(f"{reformat_csvs_path}/{file}")


# __reformat_csvs()
# new_function()
