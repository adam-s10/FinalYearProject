# https://archive.ics.uci.edu/static/public/53/iris.zip

import os
import fnmatch
import glob
import shutil
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
download_path_zips = "/Users/adam/PycharmProjects/FinalYearProject/zips/"
extraction_path_datasets = "/Users/adam/PycharmProjects/FinalYearProject/datasets/"


def list_dataset_names():
    full_dataset_list = []
    data_still_available = True
    count = 1
    while data_still_available:
        datasets = api.dataset_list(page=count, tag_ids="classification", file_type="csv")
        # if len(datasets) != 0:
        if count < 2:
            # full_dataset_list += datasets
            full_dataset_list += datasets
            count += 1
        else:
            data_still_available = False
    print(f"Number of pages visited: {count}")
    print(f"Number of datasets scraped: {len(full_dataset_list)}")
    print(f"Complete Data Set: {full_dataset_list}")
    return full_dataset_list


def download_datasets(dataset_list_as_strings):
    # Trying to only download certain files, i.e. csv's. This did not go according to plan so will probably abandon
    # Reason is that the output of the files contained in a dataset is only printed to console and returns void...
    # files_client = api.dataset_list_files_cli(dataset="joebeachcapital/30000-spotify-songs", csv_display=True)
    # files = api.dataset_list_files(dataset=dataset_list[0])
    # print(f"Files client: {files_client}")
    # print(f"Files: {files}")
    # temp2 = str(temp)
    # print(f"temp2 = {temp2}")
    # if "csv" in temp2:
    #     print("True")
    # for word in temp2:
    #     if ".csv" in word:
    # api.dataset_download_cli(dataset="joebeachcapital/30000-spotify-songs",
    #                                  path=f"{download_path}/joebeachcapital/30000-spotify-songs", unzip=True)

    for word in dataset_list_as_strings:
        path = f"{download_path_zips}{word.split('/')[1]}"
        api.dataset_download_cli(dataset=word, path=path)
    print("Download of datasets complete")
    # TODO: using something extract only the csv files (pandas/builtin?)


def extract_data(directory):
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


def list_files(directory):
    print(f"List of children: {[x[0] for x in os.walk(directory)]}")
    for file in glob.glob(f"{directory}/**/*.csv", recursive=True):
        print(file)
        shutil.copy(file, "/Users/adam/PycharmProjects/FinalYearProject/datasets/csvs")


dataList = list_dataset_names()
list_of_str = [str(x) for x in dataList]
download_datasets(list_of_str)
extract_data(list_of_str)
list_files(extraction_path_datasets)

URL = "archive.ics.edu"
URL_withClassification_andRows25 = "https://archive.ics.uci.edu/datasets?Task=Classification&skip=0&take=25&sort=desc&orderBy=NumHits&search="
URL_withClassification_andRows25_NextPage = "https://archive.ics.uci.edu/datasets?Task=Classification&skip=25&take=25&sort=desc&orderBy=NumHits&search="

# skip= will change the page, increment in 25's
# Alternatively: writing URL like so allows for everything to be displayed on 1 page,
# https://archive.ics.uci.edu/datasets?Task=Classification&skip=0&take=493&sort=desc&orderBy=NumHits&search=
# if only there was a download button from here rather than individual pages....
# kaggle does allow this but requires google authentication
# kaggle datasets list --file-type csv --tags classification -p 1 -> 97
