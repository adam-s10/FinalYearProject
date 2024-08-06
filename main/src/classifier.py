import logging
import statistics
import sys
import os
import shutil

import pandas as pd
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


# writes accuracy scores to a text file
def write_accuracy_score(file, classifier_name, scores_as_str, average):
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/classification_results (using Swift's method).txt", "a")
    f.write(f"{file},{classifier_name},{scores_as_str},{average}\n")
    f.close()


# writes any failures that occur during execution to a text file
def write_failed_files(file, error_code, error_message):
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/failed_files.txt", "a")
    f.write(f"{file},{error_code},{error_message}\n")
    f.close()


# Writes to a file to be turned into the meta-dataset (csv)
def write_meta_data(file, columns, rows, minimum, maximum, sd, mean, best_classifier):
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/metadata_file.csv", "a")
    f.write(f"{file},{columns},{rows},{minimum},{maximum},{sd},{mean},{best_classifier}\n")
    f.close()


def move_invalid_datasets():
    too_few_rows = 0
    too_few_columns = 0
    too_big = 0
    one_target_variable = 0
    fails_to_classify = 0
    for file in os.listdir(path_to_csvs):
        logger.info(f"move_invalid_datasets: File to be classified --> {file}")
        data, a, b = preprocess_data(path_to_csvs, file)

        # Return -1 if file has too few columns
        x = len(data.columns)
        if x <= 2:
            logger.warning(f"move_invalid_datasets: {file} has too few columns")
            write_failed_files(file, -1, "too few columns")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_few_columns += 1
            continue

        # If data set has less than 100 rows, return -2
        if data.shape[0] < 100:
            logger.warning(f"move_invalid_datasets: {file} has too few rows")
            write_failed_files(file, -2, "too few rows")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_few_rows += 1
            continue

        # If greater than xMB return -3
        z = os.path.getsize(f"{path_to_csvs}/{file}")
        logger.info(f"move_invalid_datasets: Size of {file} --> {z}")
        z = z / (pow(1024, 2))
        logger.info(f"move_invalid_datasets: Size of {file} in MB --> {z}")
        if z > 2:
            logger.warning(f"move_invalid_datasets: {file} is too big")
            write_failed_files(file, -3, "file is too large")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            too_big += 1
            continue

        s = b.nunique(axis="rows").to_list()[0]  # use pandas to get unique elements in each row of targets column
        if s == 1:  # If target column only contains 1 value
            logger.warning(f"move_invalid_datasets: {file} only contains 1 variable in target column")
            write_failed_files(file, -4, "only 1 variable in target column")  # write to file with code -4
            # move file to invalidDatasets
            shutil.move(f"{path_to_csvs}/{file}",
                        f"D:/PycharmProjects/FinalYearProject/invalidDatasets/{file}")
            one_target_variable += 1  # count of this failure
            continue  # move onto next file

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
            write_failed_files(file, -5, "file failed to classify")
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


# Runs all datasets in provided directory to ensure they will work for generating meta-dataset.
# Writes the results to a file to check for anomalies, such as NAN
def run_all_classifiers():
    error_count = 0
    for file in os.listdir(path_to_csvs):
        logger.info(f"File to be classified --> {file}")
        data, a, b = preprocess_data(path_to_csvs, file)

        # Classify for Support Vector Machine
        try:
            svm_classifier = svm.SVC()
            svm_acc = cross_validation(svm_classifier, a, b, "SVM", file)
        except ValueError:
            logger.error(f"{file} raised value error, skipping")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/inconsistentFailures/{file}")
            error_count += 1
            continue

        svm_st = str(svm_acc)
        svm_st = svm_st.replace("[", "")
        svm_st = svm_st.replace("]", "")
        svm_st = svm_st.replace(" ", "")
        write_accuracy_score(file, "SVM", svm_st, statistics.mean(svm_acc))
        logger.info(f"Classification for {file} using classifier SVM finished!")

        # Classify for neural network
        nn_classifier = MLPClassifier(max_iter=500)
        nn_acc = cross_validation(nn_classifier, a, b, "NN", file)

        nn_st = str(nn_acc)
        nn_st = nn_st.replace("[", "")
        nn_st = nn_st.replace("]", "")
        nn_st = nn_st.replace(" ", "")
        write_accuracy_score(file, "NN", nn_st, statistics.mean(nn_acc))
        logger.info(f"Classification for {file} using classifier NN finished!")

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        rf_acc = cross_validation(rf_classifier, a, b, "RF", file)

        rf_st = str(rf_acc)
        rf_st = rf_st.replace("[", "")
        rf_st = rf_st.replace("]", "")
        rf_st = rf_st.replace(" ", "")
        write_accuracy_score(file, "RF", rf_st, statistics.mean(rf_acc))
        logger.info(f"Classification for {file} using classifier RF finished!")

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        lr_acc = cross_validation(lr_classifier, a, b, "LR", file)

        lr_st = str(lr_acc)
        lr_st = lr_st.replace("[", "")
        lr_st = lr_st.replace("]", "")
        lr_st = lr_st.replace(" ", "")
        write_accuracy_score(file, "LR", lr_st, statistics.mean(lr_acc))
        logger.info(f"Classification for {file} using classifier LR finished!")

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        nb_acc = cross_validation(nb_classifier, a, b, "NB", file)

        nb_st = str(nb_acc)
        nb_st = nb_st.replace("[", "")
        nb_st = nb_st.replace("]", "")
        nb_st = nb_st.replace(" ", "")
        write_accuracy_score(file, "NB", nb_st, statistics.mean(nb_acc))
        logger.info(f"Classification for {file} using classifier NB finished!")

        # Once all 5 classifiers are complete move the dataset to new folder
        shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/completedDatasets/{file}")
    print("number of errors --> ", error_count)


# Cross validation method
def cross_validation(classifier, data, classes, classifier_name, file):

    logger.info(f"cross_validation: Classification for {file} using classifier {classifier_name} starting:")
    data = data.to_numpy()  # convert to numpy array
    classes = classes.values.tolist()  # convert to list
    acc = []  # create list to store accuracies
    skf = StratifiedKFold(n_splits=10, shuffle=True)  # 10-fold split with data shuffle
    skf.get_n_splits(data, classes)  # get the splits using skf
    for train_index, test_index in skf.split(data, classes):  # split the data
        # get train and test data
        train_d = data[train_index]
        train_c = [classes[j] for j in train_index]
        test_d = data[test_index]
        test_c = [classes[j] for j in test_index]

        classifier.fit(train_d, train_c)  # classify
        # Evaluate model on testing data
        y_pred = classifier.predict(test_d)
        a = accuracy_score(y_pred, test_c)  # get accuracy
        acc.append(a)  # add accuracy to list
    logger.info(f"cross_validation: Cross validation scores for {file} using classifier {classifier_name} --> {acc}")
    logger.info(f"cross_validation: Average --> {statistics.mean(acc)}")
    logger.info(f"cross_validation: Number of CV Scores used in Average --> {len(acc)}")
    return acc


# Returns the features and targets of the provided dataset
def preprocess_data(path, file):
    logger.info(f"preprocess_data: Starting for file --> {file}")
    data = pd.read_csv(f"{path}/{file}", header=None)  # Read data using pandas
    data = data.dropna()  # Remove rows with NA values
    # Split data into features and targets
    a = data.iloc[:, :-1]  # All columns except the last one, features
    b = data.iloc[:, -1:]  # Only the last column, targets
    logger.info(f"preprocess_data: Finished for file --> {file}")
    return data, a, b


# Returns standard deviation, mean, maximum, minimum of the provided accuracy
def calculate_stats(acc):
    standard_deviation = statistics.stdev(acc)
    mean = statistics.mean(acc)
    maximum = max(acc)
    minimum = min(acc)
    return standard_deviation, mean, maximum, minimum


# Creates the meta-dataset to make predictions on what the best classifier would be for a given dataset
def create_meta_dataset():
    error_count = 0
    for file in os.listdir("D:/PycharmProjects/FinalYearProject/csvDatasets"):
        data, a, b = preprocess_data(path_to_csvs, file)
        rows = data.shape[0]  # Get rows using pandas
        columns = data.shape[1]  # Get columns using pandas

        # Classify for Support Vector Machine
        try:
            svm_classifier = svm.SVC()
            svm_acc = cross_validation(svm_classifier, a, b, "SVM", file)
        except ValueError:  # If fails; skip
            logger.error(f"{file} raised value error, skipping")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/inconsistentFailures/{file}")
            error_count += 1
            continue
        svm_sd, svm_mean, svm_max, svm_min = calculate_stats(svm_acc)
        logger.info(f"create_meta_dataset: SVM returned mean --> {svm_mean}")
        logger.info(f"create_meta_dataset: SVM returned standard deviation --> {svm_sd}")
        logger.info(f"create_meta_dataset: SVM returned minimum --> {svm_min}")
        logger.info(f"create_meta_dataset: SVM returned maximum --> {svm_max}")

        # Classify for neural network
        nn_classifier = MLPClassifier(max_iter=500)
        nn_acc = cross_validation(nn_classifier, a, b, "NN", file)
        nn_sd, nn_mean, nn_max, nn_min = calculate_stats(nn_acc)
        logger.info(f"create_meta_dataset: NN returned mean --> {nn_mean}")
        logger.info(f"create_meta_dataset: NN returned standard deviation --> {nn_sd}")
        logger.info(f"create_meta_dataset: NN returned minimum --> {nn_min}")
        logger.info(f"create_meta_dataset: NN returned maximum --> {nn_max}")

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        rf_acc = cross_validation(rf_classifier, a, b, "RF", file)
        rf_sd, rf_mean, rf_max, rf_min = calculate_stats(rf_acc)
        logger.info(f"create_meta_dataset: RF returned mean --> {rf_mean}")
        logger.info(f"create_meta_dataset: RF returned standard deviation --> {rf_sd}")
        logger.info(f"create_meta_dataset: RF returned minimum --> {rf_min}")
        logger.info(f"create_meta_dataset: RF returned maximum --> {rf_max}")

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        lr_acc = cross_validation(lr_classifier, a, b, "LR", file)
        lr_sd, lr_mean, lr_max, lr_min = calculate_stats(lr_acc)
        logger.info(f"create_meta_dataset: LR returned mean --> {lr_mean}")
        logger.info(f"create_meta_dataset: LR returned standard deviation --> {lr_sd}")
        logger.info(f"create_meta_dataset: LR returned minimum --> {lr_min}")
        logger.info(f"create_meta_dataset: LR returned maximum --> {lr_max}")

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        nb_acc = cross_validation(nb_classifier, a, b, "NB", file)
        nb_sd, nb_mean, nb_max, nb_min = calculate_stats(nb_acc)
        logger.info(f"create_meta_dataset: NB returned mean --> {nb_mean}")
        logger.info(f"create_meta_dataset: NB returned standard deviation --> {nb_sd}")
        logger.info(f"create_meta_dataset: NB returned minimum --> {nb_min}")
        logger.info(f"create_meta_dataset: NB returned maximum --> {nb_max}")

        # ---Get the best classifier using mean---
        # dictionary containing means
        var = {svm_mean: "svm", nn_mean: "nn", rf_mean: "rf", lr_mean: "lr", nb_mean: "nb"}
        switch_case = var.get(max(var))  # get the value with the highest mean
        match switch_case:  # write to file depending on what was the best classifier
            case "svm":
                best_classifier = 1
                write_meta_data(file, columns, rows, svm_min, svm_max, svm_sd, svm_mean, best_classifier)
            case "nn":
                best_classifier = 2
                write_meta_data(file, columns, rows, nn_min, nn_max, nn_sd, nn_mean, best_classifier)
            case "rf":
                best_classifier = 3
                write_meta_data(file, columns, rows, rf_min, rf_max, rf_sd, rf_mean, best_classifier)
            case "lr":
                best_classifier = 4
                write_meta_data(file, columns, rows, lr_min, lr_max, lr_sd, lr_mean, best_classifier)
            case "nb":
                best_classifier = 5
                write_meta_data(file, columns, rows, nb_min, nb_max, nb_sd, nb_mean, best_classifier)

        # Once all 5 classifiers are complete move the dataset to new folder
        shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/completedDatasets/{file}")


# Predict the best classifier using created meta-dataset
def classify_metafile():
    file = "final_metadata_file.csv"
    path_to_meta = "D:/PycharmProjects/FinalYearProject/MetaDataFiles"
    f = open("D:/PycharmProjects/FinalYearProject/MetaDataFiles/results_of_meta_file.txt", "a")
    data, a, b = preprocess_data(path_to_meta, file)

    svm_results, nn_results, rf_results, lr_results, nb_results = classify_dataset(file, a, b)
    svm_sd, svm_mean, svm_max, svm_min = calculate_stats(svm_results)
    f.write(f"SVM Cross-validation: {svm_results}\n")
    f.write(f"Min: {svm_min}, Max: {svm_max}, Standard Deviation: {svm_sd}, Mean: {svm_mean}\n")

    nn_sd, nn_mean, nn_max, nn_min = calculate_stats(nn_results)
    f.write(f"NN Cross-validation: {nn_results}\n")
    f.write(f"Min: {nn_min}, Max: {nn_max}, Standard Deviation: {nn_sd}, Mean: {nn_mean}\n")

    rf_sd, rf_mean, rf_max, rf_min = calculate_stats(rf_results)
    f.write(f"RF Cross-validation: {rf_results}\n")
    f.write(f"Min: {rf_min}, Max: {rf_max}, Standard Deviation: {rf_sd}, Mean: {rf_mean}\n")

    lr_sd, lr_mean, lr_max, lr_min = calculate_stats(lr_results)
    f.write(f"LR Cross-validation: {lr_results}\n")
    f.write(f"Min: {lr_min}, Max: {lr_max}, Standard Deviation: {lr_sd}, Mean: {lr_mean}\n")

    nb_sd, nb_mean, nb_max, nb_min = calculate_stats(nb_results)
    f.write(f"NB Cross-validation: {nb_results}\n")
    f.write(f"Min: {nb_min}, Max: {nb_max}, Standard Deviation: {nb_sd}, Mean: {nb_mean}\n")
    f.close()


# runs all the classifiers returning their accuracy scores for 10-fold cross-validation
def classify_dataset(file, a, b):
    logger.info(f"run_all_classifiers: Beginning for file --> {file}")

    logger.info(f"run_all_classifiers: SVM Starting")
    svm_results = cross_validation(svm.SVC(), a, b, "SVM", file)
    logger.info(f"run_all_classifiers: SVM Finished")

    logger.info(f"run_all_classifiers: NN Starting")
    nn_results = cross_validation(MLPClassifier(max_iter=500), a, b, "NN", file)
    logger.info(f"run_all_classifiers: NN Finished")

    logger.info(f"run_all_classifiers: RF Starting")
    rf_results = cross_validation(RandomForestClassifier(), a, b, "RF", file)
    logger.info(f"run_all_classifiers: RF Finished")

    logger.info(f"run_all_classifiers: LR Starting")
    lr_results = cross_validation(LogisticRegression(), a, b, "LR", file)
    logger.info(f"run_all_classifiers: LR Finished")

    logger.info(f"run_all_classifiers: NB Starting")
    nb_results = cross_validation(GaussianNB(), a, b, "NB", file)
    logger.info(f"run_all_classifiers: NB Finished")

    logger.info(f"run_all_classifiers: Finished for file --> {file}")
    return svm_results, nn_results, rf_results, lr_results, nb_results


# move_invalid_datasets()
# run_all_classifiers()
# create_meta_dataset()
# classify_metafile()
