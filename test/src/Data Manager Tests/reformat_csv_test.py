import os.path
import shutil

import dataManagerTest as DMT

test_data_csv_files = "D:/PycharmProjects/FinalYearProject/test/test_data/reformat_csvs"
test_csv_path = "D:/PycharmProjects/FinalYearProject/test/csvDatasets"


def test_reformat():
    DMT.reformat_csvs(test_csv_path)

    number_of_files = len(os.listdir(test_csv_path))
    test1 = os.path.isfile(f"{test_csv_path}/alpha_1.csv")
    test2 = os.path.isfile(f"{test_csv_path}/alpha.csv")

    # assert number_of_files == 2
    assert test1 == True
    assert test2 == True
    # teardown()


def teardown():
    directories = os.listdir(test_csv_path)
    for file in directories:
        os.remove(f"{test_csv_path}/{file}")

    move_directories = os.listdir(test_data_csv_files)
    for file in move_directories:
        shutil.copy(f"{test_data_csv_files}/{file}", test_csv_path)
