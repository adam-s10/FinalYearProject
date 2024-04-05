import logging
import sys
import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

api = KaggleApi()
api.authenticate()

path_to_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"
csv_files = os.listdir(path_to_csvs)
failed_files = []

# Total number of files: 40599
# 3.74% or 1519 files are above 1MB
# 3.3% or 1373 files are above 2MB
# 2.6% or  1067 files are above 3MB
# 2.0% or 832 files are above 4MB
# Current iteration we are losing 3.3% of files


# Writes accuracy score of a given file name. Will return -1, -2, -3 for a variety of errors
def write_results_to_txt(file_name, value):
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/results_ignoring_above_4MB.txt", "a")
    f.write(f"{file_name},{value}\n")
    f.close()


def verify_data_classifies():
    for file in csv_files:
        logger.info(f"File to be classified --> {file}")
        data = pd.read_csv(f"{path_to_csvs}/{file}", header=None)

        # Get number of columns then remove 1 to fit array format for later processing
        # Return -1 if file has too few columns
        x = len(data.columns)
        if x <= 2:
            logger.warning(f"File has too few columns, skipping file --> {file}")
            failed_files.append(file)
            write_results_to_txt(file, -1)
            continue

        # If greater than xMB return -2
        z = os.path.getsize(f"{path_to_csvs}/{file}")
        logger.info(f"Size of file --> {z}")
        z = z / 1000000
        logger.info(f"Size of file in MB --> {z}")
        if z > 4:
            logger.warning(f"File is too big, skipping file --> {file}")
            failed_files.append(file)
            write_results_to_txt(file, -2)
            continue
        x -= 1

        # All columns except the last one
        a = data.drop(data.columns[len(data.columns) - 1], axis=1)
        # Only the last column
        b = data.drop(data.iloc[:, 0:x], axis=1)

        # Run classifier after data processing
        classifier = tree.DecisionTreeClassifier(criterion='gini')
        try:
            # If all cases pass and file classifies write accuracy to txt file
            classifier.fit(a, b)
            # Evaluate model on testing data
            y_pred = classifier.predict(a)
            acc = accuracy_score(y_pred, b)
            logger.info(f"Accuracy score for file {file} returned --> {acc}")
            write_results_to_txt(file, acc)
        except:
            # If classifier fails return -3
            logger.warning(f"Something went wrong with file --> {file}")
            failed_files.append(file)
            write_results_to_txt(file, -3)


def move_invalid_datasets():
    for file in csv_files:
        logger.info(f"File to be classified --> {file}")
        data = pd.read_csv(f"{path_to_csvs}/{file}", header=None)

        # Get number of columns then remove 1 to fit array format for later processing
        # Return -1 if file has too few columns
        x = len(data.columns)
        if x <= 2:
            logger.warning(f"File has too few columns, skipping file --> {file}")
            failed_files.append(file)
            # write_results_to_txt(file, -1)
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            continue

        # If greater than xMB return -2
        z = os.path.getsize(f"{path_to_csvs}/{file}")
        logger.info(f"Size of file --> {z}")
        z = z / 1000000
        logger.info(f"Size of file in MB --> {z}")
        if z > 2:
            logger.warning(f"File is too big, skipping file --> {file}")
            failed_files.append(file)
            # write_results_to_txt(file, -2)
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            continue
        x -= 1

        # All columns except the last one
        a = data.drop(data.columns[len(data.columns) - 1], axis=1)
        # Only the last column
        b = data.drop(data.iloc[:, 0:x], axis=1)

        # Run classifier after data processing
        classifier = tree.DecisionTreeClassifier(criterion='gini')
        try:
            # If all cases pass and file classifies do nothing
            classifier.fit(a, b)
            # Evaluate model on testing data
            y_pred = classifier.predict(a)
            acc = accuracy_score(y_pred, b)
            logger.info(f"Accuracy score for file {file} returned --> {acc}")
        except:
            # If classifier fails move to new directory
            logger.warning(f"Something went wrong with file --> {file}")
            failed_files.append(file)
            # write_results_to_txt(file, -3)
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")


def get_tags_for_csvs():
    for file in csv_files:
        x = file[8:]
        logger.info(f"CSV file to find tags --> {x}")

        datasets = api.dataset_list(search=x, file_type="csv", page=1)
        for i in range(0, len(datasets)):
            d = datasets[i]
            if len(d.tags) == 0:
                f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/tags_list2.txt", "a")
                f.write(f"{x},{d.ref},none\n")
                f.close()
            else:
                for t in d.tags:
                    logger.info(f"{x}, {str(d.ref)}, {str(t)}")
                    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/tags_list2.txt", "a")
                    f.write(f"{x},{d.ref},{t}\n")
                    f.close()


def run_all_classifiers():
    for file in csv_files:
        logger.info(f"File to be classified --> {file}")

        # Preprocess the data
        data = pd.read_csv(f"{path_to_csvs}/{file}", header=None)
        x = len(data.columns)
        x -= 1

        # Prepare data
        # All columns except the last one
        a = data.drop(data.columns[len(data.columns) - 1], axis=1)
        # Only the last column
        b = data.drop(data.iloc[:, 0:x], axis=1)
        #Filtering
        s = set(b)
        print(s)
        if len(s) == 1:
            logger.warning(f"{file} only contains 1 variable in target column, skipping")
            write_accuracy_score(file, "N/A", -2)
            continue
        if data.shape[0] < 100:
            logger.warning(f"{file} has too few rows, skipping")
            write_accuracy_score(file, "N/A", -3)
            continue

        # Classify for Support Vector Machine
        svm_classifier = svm.SVC(kernel="linear")
        cross_validation(svm_classifier, a, b, "SVM", file)

        # Classify for neural network
        nn_classifier = MLPClassifier()
        cross_validation(nn_classifier, a, b, "NN", file)

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        cross_validation(rf_classifier, a, b, "RF", file)

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        cross_validation(lr_classifier, a, b, "LR", file)

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        cross_validation(nb_classifier, a, b, "NB", file)


def cross_validation(classifier, a, b, classifier_name, file):
    acc = []
    try:
        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(a, b)
        for train_index, test_index in skf.split(a, b):
            train_d = a[train_index]
            train_c = [b[j] for j in train_index]
            test_d = a[test_index]
            test_c = [b[j] for j in test_index]
            classifier.fit(train_d, train_c)

            # Evaluate model on testing data
            y_pred = classifier.predict(test_d)
            acc = accuracy_score(y_pred, test_c)
        write_accuracy_score(file, classifier_name, acc)
    except:
        logger.info("An error occurred")
        write_accuracy_score(file, classifier_name, -1)


def write_accuracy_score(file, classifier_name, acc):
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/classification_results3.txt", "a")
    a = str(acc)
    a = a.replace("[", "")
    a = a.replace("]", "")
    f.write(f"{file},{classifier_name},{a}\n")
    f.close()


# get_tags_for_csvs()
# move_invalid_datasets()
run_all_classifiers()
