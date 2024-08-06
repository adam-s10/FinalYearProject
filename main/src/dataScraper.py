import os
import statistics
import zipfile
from pathlib import Path
import deprecation

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

download_path_zips = "D:/PycharmProjects/FinalYearProject/zips/"
extraction_path_datasets = "D:/PycharmProjects/FinalYearProject/extractedZips"
path_to_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"
csv_files = os.listdir(path_to_csvs)


# Gathers the information needed to interact with api.download_datasets_cli
# Hits Kaggle API until no more datasets are returned for the given page
def list_dataset_names():
    full_dataset_list = []
    data_still_available = True
    page_number = 1
    while data_still_available:  # while kaggle keeps returning datasets
        # call to kaggle to return all datasets with tag classification and file type csv
        datasets = api.dataset_list(page=page_number, tag_ids="classification", file_type="csv")
        if len(datasets) != 0:
            full_dataset_list += datasets  # append file location to list
            page_number += 1  # add 1 to get next page of datasets
        else:
            data_still_available = False
    # log lines providing info related to method's run
    logger.info(f"list_dataset_names: Number of pages visited --> {page_number}")
    logger.info(f"list_dataset_names: Number of extractedZips scraped --> {len(full_dataset_list)}")
    print(f"\n Complete Data Set: {full_dataset_list}")
    return full_dataset_list


# Uses list generated from list_dataset_names to download the identified datasets to a given location
def download_datasets(dataset_list_as_strings):
    logger.info("download_datasets: Download Starting!")
    for word in dataset_list_as_strings:
        path = f"{download_path_zips}{word.split('/')[1]}"  # creates a new directory based off file name downloaded
        try:
            api.dataset_download_cli(dataset=word, path=path)  # download client for kaggle api
        except:
            logger.info(f"This data set could not be scraped --> {download_path_zips}{word.split('/')[1]}")
    logger.info("download_datasets: Download Complete!")


# Method to extract all zip files downloaded from kaggle api
@deprecation.deprecated(deprecated_in="Sprint 2",
                        details="Dangerous when handling large amount of files to be unzipped (potential Zip Bomb). "
                                "Use extract_data in dataManager.py instead.")
def extract_data(directory):
    extracted_locations = []
    for word in directory:  # for each directory in the parent directory
        file_name = word.split('/')[1]  # separate to get file name
        zipfile_location = f"{download_path_zips}{file_name}"
        extracted_to = f"{extraction_path_datasets}{file_name}"  # new directory for data to be extracted to
        path = Path(zipfile_location)
        for i in path.glob("*.zip"):  # for each file that is a zip
            with zipfile.ZipFile(i, mode="r") as Zip:
                Zip.extractall(extracted_to)  # extract the file to the provided location
        extracted_locations.append(file_name)
    print(extracted_locations)  # print successful extractions


# Get the tags related to a dataset using kaggle api
def get_tags_for_csvs():
    for file in csv_files:
        x = file[8:]  # use indexing to remove the updated_ from the name of the file returned by os listdir
        logger.info(f"CSV file to find tags --> {x}")

        datasets = api.dataset_list(search=x, file_type="csv", page=1)  # calls kaggle api for info on dataset
        if len(datasets) == 0:
            logger.info(f"No results found for --> {x}")
        for i in range(0, len(datasets)):  # for each result
            d = datasets[i]
            if len(d.tags) == 0:  # if no tags found write none in file
                f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/missing_tags_list.txt", "a")
                f.write(f"{x},{d.ref},none\n")
                f.close()
            else:
                for t in d.tags:  # for each tag returned write file name, reference and the tag
                    logger.info(f"{x}, {str(d.ref)}, {str(t)}")
                    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/missing_tags_list.txt", "a")
                    f.write(f"{x},{d.ref},{t}\n")
                    f.close()


# Finds the model tag related to a dataset name to estimate the dataset tag
# Prints results on a new line to be copied into excel
def find_modal_tag():
    # open metadata file in read more
    metadata_file = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/metadata_file.csv", "r")
    for meta_line in metadata_file:  # for each line in metadata file
        meta_line = meta_line.split(",")  # split the features
        x = meta_line[0][8:]  # get dataset name and remove updated_
        arr = []  # empty list to hold the tags

        # open tags file in read mode
        tags_file = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/tags_list_final.csv", "r")
        for tags_line in tags_file:  # for each line in tag file
            if x in tags_line:  # if dataset name is in tags file
                y = tags_line.split(",")[4]  # get each tag
                arr.append(y)  # add each tag for the file to list
        tags_file.close()
        if len(arr) != 0:  # if the filename from metadata file exists in tags file
            print(statistics.mode(arr))  # calculate the modal tag
        else:
            print(0)  # assume tag = 0 if nothing found


# list_of_datasets = [str(x) for x in __list_dataset_names()]
# __download_datasets(list_of_datasets)

get_tags_for_csvs()
