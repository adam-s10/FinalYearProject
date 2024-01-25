import os
import logging
import sys
import time
import shutil

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
