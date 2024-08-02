import logging
import statistics
import sys
import os
import shutil
import pandas as pd
import unittest

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
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

path_to_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"
csv_files = os.listdir(path_to_csvs)


def get_shape():
    data = pd.read_csv(f"{path_to_csvs}/updated_01-01-02-01-02-02-14.csv", header=None)
    data = data.dropna()
    print(f"Number of rows for file updated_01-01-02-01-02-02-14.csv --> {data.shape[0]}")
    print(f"Number of columns for updated_01-01-02-01-02-02-14.csv --> {data.shape[1]}")


def preprocess_data(file):
    logger.info(f"preprocess_data: Starting for file --> {file}")
    data = pd.read_csv(f"{path_to_csvs}/{file}", header=None)  # Read data using pandas
    data = data.dropna()  # Remove rows with NA values
    # Split data into classes and targets
    a = data.iloc[:, :-1]  # All columns except the last one, classes
    b = data.iloc[:, -1:]  # Only the last column, targets
    logger.info(f"preprocess_data: Finished for file --> {file}")
    return data, a, b


def move_invalid_datasets():
    too_few_rows = 0
    too_few_columns = 0
    too_big = 0
    one_target_variable = 0
    fails_to_classify = 0
    for file in os.listdir(path_to_csvs):
        logger.info(f"move_invalid_datasets: File to be checked --> {file}")
        data, a, b = preprocess_data(file)

        # Return -1 if file has too few columns
        x = len(data.columns)
        if x <= 2:
            logger.warning(f"move_invalid_datasets: {file} has too few columns")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_few_columns += 1
            continue

        # If data set has less than 100 rows, return -2
        if data.shape[0] < 100:
            logger.warning(f"move_invalid_datasets: {file} has too few rows")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_few_rows += 1
            continue

        # If greater than xMB return -3
        z = os.path.getsize(f"{path_to_csvs}/{file}")
        logger.info(f"move_invalid_datasets: Size of {file} --> {z}")
        z = z / (pow(1000, 2))
        logger.info(f"move_invalid_datasets: Size of {file} in MB --> {z}")
        if z > 2:
            logger.warning(f"move_invalid_datasets: {file} is too big")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_big += 1
            continue

        # If target column only contains 1 value, return -4
        s = b.nunique(axis="rows").to_list()[0]
        if s == 1:
            logger.warning(f"move_invalid_datasets: {file} only contains 1 variable in target column")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            one_target_variable += 1
            continue

        # Run classifier after data processing
        classifier = tree.DecisionTreeClassifier(criterion='gini')
        try:
            # If all cases pass and file classifies do nothing
            classifier.fit(a, b.values.ravel())
            # Evaluate model on testing data
            y_pred = classifier.predict(a)
            acc = accuracy_score(y_pred, b.values.ravel())
            logger.info(f"move_invalid_datasets: Accuracy score for {file} returned --> {acc}")
        except:
            # If classifier fails, return -5
            logger.warning(f"move_invalid_datasets: Something went wrong with --> {file}")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            fails_to_classify += 1

    print("")
    total_invalid = too_few_rows + too_few_columns + too_big + one_target_variable + fails_to_classify
    logger.info(f"move_invalid_datasets: Total Number of files --> {len(csv_files)}")
    logger.info(f"move_invalid_datasets: Total number of invalid files --> {total_invalid}")
    logger.info(f"move_invalid_datasets: Number of files with too few rows --> {too_few_rows}")
    logger.info(f"move_invalid_datasets: Number of files with too few columns --> {too_few_columns}")
    logger.info(f"move_invalid_datasets: Number of files which were too big  --> {too_big}")
    logger.info(f"move_invalid_datasets: Number of files with only 1 variable in target column --> "
                f"{one_target_variable}")
    logger.info(f"move_invalid_datasets: Number of files which failed to classify --> {fails_to_classify}")


def run_all_classifiers():
    for file in os.listdir(path_to_csvs):
        logger.info(f"File to be classified --> {file}")

        data, a, b = preprocess_data(file)

        # Classify for Support Vector Machine
        # try:
        svm_classifier = svm.SVC()
        svm_acc = cross_validation(svm_classifier, a, b, "SVM", file)
        # except ValueError:
        logger.info(f"Classification for {file} using classifier SVM finished!")

        # Classify for neural network
        nn_classifier = MLPClassifier(max_iter=500)
        nn_acc = cross_validation(nn_classifier, a, b, "NN", file)
        logger.info(f"Classification for {file} using classifier NN finished!")

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        rf_acc = cross_validation(rf_classifier, a, b, "RF", file)
        logger.info(f"Classification for {file} using classifier RF finished!")

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        lr_acc = cross_validation(lr_classifier, a, b, "LR", file)
        logger.info(f"Classification for {file} using classifier LR finished!")

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        nb_acc = cross_validation(nb_classifier, a, b, "NB", file)


def cross_validation(classifier, data, classes, classifier_name, file):

    logger.info(f"cross_validation: Classification for {file} using classifier {classifier_name} starting:")
    data = data.to_numpy()
    classes = classes.values.tolist()
    acc = []
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    skf.get_n_splits(data, classes)
    for train_index, test_index in skf.split(data, classes):
        train_d = data[train_index]
        train_c = [classes[j] for j in train_index]
        test_d = data[test_index]
        test_c = [classes[j] for j in test_index]

        classifier.fit(train_d, train_c)
        # Evaluate model on testing data
        y_pred = classifier.predict(test_d)
        a = accuracy_score(y_pred, test_c)
        acc.append(a)
    logger.info(f"cross_validation: Cross validation scores for {file} using classifier {classifier_name} --> {acc}")
    logger.info(f"cross_validation: Average --> {statistics.mean(acc)}")
    logger.info(f"cross_validation: Number of CV Scores used in Average --> {len(acc)}")
    return acc


# move_invalid_datasets()
# run_all_classifiers()
get_shape()
