import os
import logging
import sys
import time
import shutil

import deprecation
import pandas as pd
from pathlib import Path
from classifier import preprocess_data, verify_data_classifies, FileManager
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
                                "Use get_list_of_csvs in data_manager.py instead")
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


# If Number of columns are less than or equal to min_columns return True
def check_too_few_columns(data, file_name, min_columns=2):
    number_of_columns = len(data.columns)
    if number_of_columns <= min_columns:
        logger.warning(f"check_too_few_columns: {file_name} has below the minimum number of columns ({min_columns})")
        return True
    return False


# If number of rows is less than min_rows return True
def check_too_few_rows(data, file_name, min_rows=100):
    number_of_rows = data.shape[0]
    if number_of_rows < min_rows:
        logger.warning(f"check_too_few_rows: {file_name} has below the minimum number of rows ({min_rows})")
        return True
    return False


# If file size is greater than max_size (MiB) return True
def check_file_too_large(path, file_name, max_size=2):
    file_size = os.path.getsize(path)
    logger.info(f"check_too_large: Size of {file_name} --> {file_size}")
    file_size_mib = file_size / (1024 ** 2)  # convert file from bytes to MiB
    logger.info(f"check_too_large: Size of {file_name} in MiB --> {file_size_mib}")
    if file_size_mib > max_size:
        logger.warning(f"check_too_large: {file_name} is greater than the max size ({max_size} MiB)")
        return True
    return False


# If file has only 1 variable in target column return True
def check_more_than_one_element(targets, file):
    # use pandas to get unique elements in each row of targets column
    number_of_unique_variables = targets.nunique(axis="rows").to_list()[0]
    if number_of_unique_variables == 1:  # If target column only contains 1 value
        logger.warning(f"check_more_than_one_element: {file} only contains 1 variable in target column")
        return True
    return False


# Move a file from source to destination using shutil
def move_invalid_datasets(source, destination='D:/PycharmProjects/FinalYearProject/invalidDatasets/'):
    shutil.move(source, destination)


# writes any failures that occur during execution to a text file
def write_failed_files(file, error_code, error_message):
    with FileManager('failed_files', 'a',
                     file_root='D:/PycharmProjects/FinalYearProject/MetaDataFiles') as f:
        f.write(f"{file},{error_code},{error_message}\n")


def validate_datasets(directory_root):
    too_few_rows, too_few_columns, too_big, one_target_variable, fails_to_classify = (0,) * 5
    total_number_of_files = len(os.listdir(directory_root))
    for file in os.listdir(directory_root):
        logger.info(f"validate_data_sets: File to be validated --> {file}")
        dataset, data, classes = preprocess_data(directory_root, file)
        file_root = f"{directory_root}/{file}"

        # Move dataset with too few columns and write error code (-1) to file
        if check_too_few_columns(dataset, file):
            write_failed_files(file, -1, "too few columns")
            move_invalid_datasets(file_root)
            too_few_columns += 1
            continue

        # Move dataset with too few rows and write error code (-2) to file
        if check_too_few_rows(dataset, file):
            write_failed_files(file, -2, "too few rows")
            move_invalid_datasets(file_root)
            too_few_rows += 1
            continue

        # Move dataset that is too large and write error code (-3) to file
        if check_file_too_large(file_root, file):
            write_failed_files(file, -3, "file is too large")
            move_invalid_datasets(file_root)
            too_big += 1
            continue

        # Move dataset with only 1 unique variable in target column and write error code (-4) to file
        if check_more_than_one_element(classes, file):
            write_failed_files(file, -4, "only 1 variable in target column")
            move_invalid_datasets(file_root)
            one_target_variable += 1
            continue

        # Move dataset that does not classify and write error code (-5) to file
        if not verify_data_classifies(data, classes, file):
            write_failed_files(file, -5, "file failed to classify")
            move_invalid_datasets(file_root)
            fails_to_classify += 1

    print("")
    total_invalid = too_few_rows + too_few_columns + too_big + one_target_variable + fails_to_classify
    logger.info(f"validate_datasets: Total number of files checked --> {total_number_of_files}")
    logger.info(f"validate_datasets: Total number of invalid files --> {total_invalid}")
    logger.info(f"validate_datasets: Number of files with too few rows --> {too_few_rows}")
    logger.info(f"validate_datasets: Number of files with too few columns --> {too_few_columns}")
    logger.info(f"validate_datasets: Number of files which were too big  --> {too_big}")
    logger.info(f"validate_datasets: Number of files with only 1 variable in target column --> {one_target_variable}")
    logger.info(f"validate_datasets: Number of files which failed to classify --> {fails_to_classify}")


# extract_data()
# reformat_csvs(test_csv_reformatting)
# extract_data(test_csv_reformatting_live_data)
extract_data(download_path_zips)

