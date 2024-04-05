import os
import glob
import shutil
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import logging
import sys

# Get Kaggle API Ready
api = KaggleApi()
api.authenticate()

# Instantiate Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# download_path_zips = "/Users/adam/PycharmProjects/FinalYearProject/zips/"
download_path_zips = "D:/PycharmProjects/FinalYearProject/zips/"
# extraction_path_datasets = "/Users/adam/PycharmProjects/FinalYearProject/extractedZips/"
extraction_path_datasets = "D:/PycharmProjects/FinalYearProject/extractedZips"

# TODO: delete as part of reformatting
# def download_data():
#     __download_datasets(list_of_datasets)
#     zip_cleanup_service()
#
#
# def zip_cleanup_service():
#     __extract_data(list_of_datasets)
#     __delete_unused_files(download_path_zips)


# Gathers the information needed to interact with api.download_datasets_cli
# Hits Kaggle API until no more datasets are returned for the given page - count
def __list_dataset_names():
    full_dataset_list = []
    data_still_available = True
    count = 1
    while data_still_available:
        datasets = api.dataset_list(page=count, tag_ids="classification", file_type="csv")
        if len(datasets) != 0:
        # if count < 2: # Used only to test
            full_dataset_list += datasets
            count += 1
        else:
            data_still_available = False
    logger.info(f"list_dataset_names: Number of pages visited --> {count}")
    logger.info(f"list_dataset_names: Number of extractedZips scraped --> {len(full_dataset_list)}")
    print(f"/n Complete Data Set: {full_dataset_list}")
    return full_dataset_list


# Uses list generate from __list_dataset_names to download the identified datasets to a given location
def __download_datasets(dataset_list_as_strings):
    logger.info("download_datasets: Download Starting!")
    for word in dataset_list_as_strings:
        path = f"{download_path_zips}{word.split('/')[1]}"
        try:
            api.dataset_download_cli(dataset=word, path=path)
        except:
            logger.info(f"This data set could not be scraped --> {download_path_zips}{word.split('/')[1]}")
    logger.info("download_datasets: Download Complete!")


# TODO: delete these as part of reformatting
def __extract_data(directory):
    extracted_locations = []
    for word in directory:
        file_name = word.split('/')[1]
        zipfile_location = f"{download_path_zips}{file_name}"
        extracted_to = f"{extraction_path_datasets}{file_name}"
        path = Path(zipfile_location)
        for i in path.glob("*.zip"):
            with zipfile.ZipFile(i, "r") as Zip:
                Zip.extractall(extracted_to)
        extracted_locations.append(file_name)
    print(extracted_locations)
#
#
# def __delete_unused_files(path):
#     logger.warning(f"This directory will be DELETED in path --> {path}, in 10 seconds...")
#     time.sleep(10)
#     shutil.rmtree(path)
#     logger.info(f"Deletion complete in path --> {path}")


list_of_datasets = [str(x) for x in __list_dataset_names()]
__download_datasets(list_of_datasets)
