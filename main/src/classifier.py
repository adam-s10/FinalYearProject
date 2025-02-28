import logging
import statistics
import sys
import os
from os.path import dirname, abspath
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


# Class providing a context manager and benefits to file manipulation and generation for this project
class FileManager:

    # Assumes file is located or is to be generated in the same directory as python file; unless otherwise specified
    def __init__(self, file_name, mode, extension='.txt', file_root=dirname(abspath(__file__))):
        self.file_name = file_name
        self.mode = mode
        self.extension = extension
        self.file_root = file_root

    def __enter__(self):
        self.file = open(self.file_root + '\\' + self.file_name + self.extension, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_val is not None:
            logger.error(f'FileManager for {self.file_name}: Encountered exception --> {exc_val}')
            logger.error(f'FileManager for {self.file_name}: Exception type --> {exc_type}')
            return True

    def __repr__(self):
        return f'FileManager for {self.file_name}: \nWorking in mode --> {self.mode}\nWith root --> {self.file_root}'


class Classifier:

    def __init__(self, classifier, classifier_name):
        if not isinstance(classifier_name, str):
            raise ValueError('Variable classifier_name should be of type str')

        self.classifier = classifier
        self.classifier_name = classifier_name
        self.accuracies = []

    def cross_validation(self, data, classes, file, n_splits=10):
        logger.info(f"cross_validation: Classification for {file} using classifier {self.classifier_name} starting:")
        data = data.to_numpy()  # convert to numpy array
        classes = classes.values.tolist()  # convert to list
        # acc = []  # create list to store accuracies
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)  # n-fold split with data shuffle
        skf.get_n_splits(data, classes)  # get the splits using skf
        for train_index, test_index in skf.split(data, classes):  # split the data
            # get train and test data
            train_d = data[train_index]
            train_c = [classes[j] for j in train_index]
            test_d = data[test_index]
            test_c = [classes[j] for j in test_index]

            self.classifier.fit(train_d, train_c)  # classify
            # Evaluate model on testing data
            y_pred = self.classifier.predict(test_d)
            a = accuracy_score(y_pred, test_c)  # get accuracy
            self.accuracies.append(a)  # add accuracy to list
        logger.info(f"cross_validation: Cross validation scores for {file} using classifier {self.classifier_name} "
                    f"--> {self.accuracies}")
        logger.info(f"cross_validation: Average --> {statistics.mean(self.accuracies)}")
        logger.info(f"cross_validation: Number of CV Scores used in Average --> {len(self.accuracies)}")
        return self.accuracies

    # Uses values stored in self.accuracies (generated using cross_validation method) to calculate standard deviation,
    # mean, maximum and minimum for those accuracies
    def calculate_stats(self):
        standard_deviation = statistics.stdev(self.accuracies)
        mean = statistics.mean(self.accuracies)
        maximum = max(self.accuracies)
        minimum = min(self.accuracies)
        return float(standard_deviation), float(mean), float(maximum), float(minimum)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.accuracies):
            raise StopIteration
        else:
            accuracy = self.accuracies[self._index]
            self._index += 1
            return accuracy

    def __repr__(self):
        return f'Classifier of type {self.classifier_name}'


# Tests that a dataset can classify with a simple decision tree
def verify_data_classifies(data, classes, file):
    classifier = tree.DecisionTreeClassifier(criterion='gini')
    try:
        classifier.fit(data, classes.values.ravel())
    except ValueError:
        logger.warning(f"verify_data_classifies: ValueError raised for --> {file}")
        return None
    # Evaluate model on testing data
    y_pred = classifier.predict(data)
    acc = accuracy_score(y_pred, classes.values.ravel())
    logger.info(f"verify_data_classifies: Accuracy score for {file} returned --> {acc}")
    return acc


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

path_to_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"


# writes accuracy scores to a text file
def write_accuracy_score(file, classifier_name, scores_as_str, average):
    with FileManager('classification_results', 'a',
                     file_root='D:/PycharmProjects/FinalYearProject/MetaDataFiles') as f:
        f.write(f"{file},{classifier_name},{scores_as_str},{average}\n")


# writes any failures that occur during execution to a text file
def write_failed_files(file, error_code, error_message):
    with FileManager('failed_files', 'a',
                     file_root='D:/PycharmProjects/FinalYearProject/MetaDataFiles') as f:
        f.write(f"{file},{error_code},{error_message}\n")


# Writes to a file to be turned into the meta-dataset (csv)
def write_meta_data(file, columns, rows, minimum, maximum, sd, mean, best_classifier):
    with FileManager('metadata_file3', 'a', '.csv',
                     'D:/PycharmProjects/FinalYearProject/MetaDataFiles') as f:
        f.write(f"{file},{columns},{rows},{minimum},{maximum},{sd},{mean},{best_classifier}\n")


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


# TODO: consider moving to dataManager.py -
def validate_datasets(directory_root):
    too_few_rows, too_few_columns, too_big, one_target_variable, fails_to_classify = (0,) * 5
    total_number_of_files = len(os.listdir(directory_root))
    for file in os.listdir(directory_root):
        logger.info(f"validate_data_sets: File to be validated --> {file}")
        dataset, data, classes = preprocess_data(directory_root, file)
        file_root = f"{directory_root}/{file}"

        # Move dataset with too few columns and write error code (-1) to file
        if check_too_few_columns(dataset, file):
            # write_failed_files(file, -1, "too few columns")
            move_invalid_datasets(file_root)
            too_few_columns += 1
            continue

        # Move dataset with too few rows and write error code (-2) to file
        if check_too_few_rows(dataset, file):
            # write_failed_files(file, -2, "too few rows")
            move_invalid_datasets(file_root)
            too_few_rows += 1
            continue

        # Move dataset that is too large and write error code (-3) to file
        if check_file_too_large(file_root, file):
            # write_failed_files(file, -3, "file is too large")
            move_invalid_datasets(file_root)
            too_big += 1
            continue

        # Move dataset with only 1 unique variable in target column and write error code (-4) to file
        if check_more_than_one_element(classes, file):
            # write_failed_files(file, -4, "only 1 variable in target column")
            move_invalid_datasets(file_root)
            one_target_variable += 1
            continue

        # Move dataset that does not classify and write error code (-5) to file
        if not verify_data_classifies(data, classes, file):
            # write_failed_files(file, -5, "file failed to classify")
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


# Runs all datasets in provided directory to ensure they will work for generating meta-dataset.
# Writes the results to a file to check for anomalies, such as NAN
def run_all_classifiers():
    error_count = 0
    for file in os.listdir(path_to_csvs):
        logger.info(f"File to be classified --> {file}")
        dataset, data, classes = preprocess_data(path_to_csvs, file)

        # Classify for Support Vector Machine
        try:
            svm_classifier = Classifier(svm.SVC(), 'Support Vector Machine')
            svm_acc = svm_classifier.cross_validation(data, classes, file)
        except ValueError:
            logger.error(f"{file} raised value error, skipping")
            shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/inconsistentFailures/{file}")
            error_count += 1
            continue

        svm_acc_str = ','.join(map(str, svm_acc))
        write_accuracy_score(file, "SVM", svm_acc_str, statistics.mean(svm_acc))
        logger.info(f"Classification for {file} using classifier SVM finished!")

        # Classify for Neural Network
        nn_classifier = Classifier(MLPClassifier(max_iter=500), 'Neural Network')
        nn_acc = nn_classifier.cross_validation(data, classes, file)

        nn_acc_str = ','.join(map(str, nn_acc))
        write_accuracy_score(file, "NN", nn_acc_str, statistics.mean(nn_acc))
        logger.info(f"Classification for {file} using classifier NN finished!")

        # Classify for Random Forest
        rf_classifier = Classifier(RandomForestClassifier(), 'Random Forest')
        rf_acc = rf_classifier.cross_validation(data, classes, file)

        rf_acc_str = ','.join(map(str, rf_acc))
        write_accuracy_score(file, "RF", rf_acc_str, statistics.mean(rf_acc))
        logger.info(f"Classification for {file} using classifier RF finished!")

        # Classify for Logistic Regression
        lr_classifier = Classifier(LogisticRegression(), 'Logistic Regression')
        lr_acc = lr_classifier.cross_validation(data, classes, file)

        lr_acc_str = ','.join(map(str, lr_acc))
        write_accuracy_score(file, "LR", lr_acc_str, statistics.mean(lr_acc))
        logger.info(f"Classification for {file} using classifier LR finished!")

        # Classify for Naive Bayes
        nb_classifier = Classifier(GaussianNB(), 'Naive Bayes')
        nb_acc = nb_classifier.cross_validation(data, classes, file)

        nb_acc_str = ','.join(map(str, nb_acc))
        write_accuracy_score(file, "NB", nb_acc_str, statistics.mean(nb_acc))
        logger.info(f"Classification for {file} using classifier NB finished!")
    print("number of errors --> ", error_count)


# Runs all datasets in provided directory to ensure they will work for generating meta-dataset.
# Writes the results to a file to check for anomalies, such as NAN
# TODO: verify functionality remains the same - Done
# TODO: Remove duplicate code - remove by 03/05/25
def run_all_classifiers_old():
    error_count = 0
    for file in os.listdir(path_to_csvs):
        logger.info(f"File to be classified --> {file}")
        dataset, data, classes = preprocess_data(path_to_csvs, file)

        # Classify for Support Vector Machine
        try:
            svm_classifier = svm.SVC()
            svm_acc = cross_validation(svm_classifier, data, classes, "SVM", file)
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
        nn_acc = cross_validation(nn_classifier, data, classes, "NN", file)

        nn_st = str(nn_acc)
        nn_st = nn_st.replace("[", "")
        nn_st = nn_st.replace("]", "")
        nn_st = nn_st.replace(" ", "")
        write_accuracy_score(file, "NN", nn_st, statistics.mean(nn_acc))
        logger.info(f"Classification for {file} using classifier NN finished!")

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        rf_acc = cross_validation(rf_classifier, data, classes, "RF", file)

        rf_st = str(rf_acc)
        rf_st = rf_st.replace("[", "")
        rf_st = rf_st.replace("]", "")
        rf_st = rf_st.replace(" ", "")
        write_accuracy_score(file, "RF", rf_st, statistics.mean(rf_acc))
        logger.info(f"Classification for {file} using classifier RF finished!")

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        lr_acc = cross_validation(lr_classifier, data, classes, "LR", file)

        lr_st = str(lr_acc)
        lr_st = lr_st.replace("[", "")
        lr_st = lr_st.replace("]", "")
        lr_st = lr_st.replace(" ", "")
        write_accuracy_score(file, "LR", lr_st, statistics.mean(lr_acc))
        logger.info(f"Classification for {file} using classifier LR finished!")

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        nb_acc = cross_validation(nb_classifier, data, classes, "NB", file)

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
# TODO: Remove duplicate code - remove by 03/01/25
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
    dataset = pd.read_csv(f"{path}/{file}", header=None)  # Read data using pandas
    dataset = dataset.dropna()  # Remove rows with NA values
    # Split data into features and targets
    data = dataset.iloc[:, :-1]  # All columns except the last one, features
    classes = dataset.iloc[:, -1:]  # Only the last column, targets
    logger.info(f"preprocess_data: Finished for file --> {file}")
    return dataset, data, classes


# Returns standard deviation, mean, maximum, minimum of the provided accuracy
# TODO: Verify functionality remains the same
# TODO: Remove duplicate code - remove by 03/01/25
def calculate_stats(acc):
    standard_deviation = statistics.stdev(acc)
    mean = statistics.mean(acc)
    maximum = max(acc)
    minimum = min(acc)
    return standard_deviation, mean, maximum, minimum


# Creates the meta-dataset to make predictions on what the best classifier would be for a given dataset
def create_meta_dataset():
    for file in os.listdir("D:/PycharmProjects/FinalYearProject/csvDatasets"):
        dataset, data, classes = preprocess_data(path_to_csvs, file)
        rows = float(dataset.shape[0])  # Get rows using pandas
        columns = float(dataset.shape[1])  # Get columns using pandas

        # Classify for Support Vector Machine
        svm_classifier = Classifier(svm.SVC(), 'Support Vector Machine')
        svm_classifier.cross_validation(data, classes, file)
        svm_sd, svm_mean, svm_max, svm_min = svm_classifier.calculate_stats()
        logger.info(f"create_meta_dataset: SVM returned mean --> {svm_mean}")
        logger.info(f"create_meta_dataset: SVM returned standard deviation --> {svm_sd}")
        logger.info(f"create_meta_dataset: SVM returned minimum --> {svm_min}")
        logger.info(f"create_meta_dataset: SVM returned maximum --> {svm_max}")

        # Classify for neural network
        nn_classifier = Classifier(MLPClassifier(max_iter=500), 'Neural Network')
        nn_classifier.cross_validation(data, classes, file)
        nn_sd, nn_mean, nn_max, nn_min = nn_classifier.calculate_stats()
        logger.info(f"create_meta_dataset: NN returned mean --> {nn_mean}")
        logger.info(f"create_meta_dataset: NN returned standard deviation --> {nn_sd}")
        logger.info(f"create_meta_dataset: NN returned minimum --> {nn_min}")
        logger.info(f"create_meta_dataset: NN returned maximum --> {nn_max}")

        # Classify for random forrest
        rf_classifier = Classifier(RandomForestClassifier(), 'Random Forest')
        rf_classifier.cross_validation(data, classes, file)
        rf_sd, rf_mean, rf_max, rf_min = rf_classifier.calculate_stats()
        logger.info(f"create_meta_dataset: RF returned mean --> {rf_mean}")
        logger.info(f"create_meta_dataset: RF returned standard deviation --> {rf_sd}")
        logger.info(f"create_meta_dataset: RF returned minimum --> {rf_min}")
        logger.info(f"create_meta_dataset: RF returned maximum --> {rf_max}")

        # Classify for Logistic Regression
        lr_classifier = Classifier(LogisticRegression(), 'Logistic Regression')
        lr_classifier.cross_validation(data, classes, file)
        lr_sd, lr_mean, lr_max, lr_min = lr_classifier.calculate_stats()
        logger.info(f"create_meta_dataset: LR returned mean --> {lr_mean}")
        logger.info(f"create_meta_dataset: LR returned standard deviation --> {lr_sd}")
        logger.info(f"create_meta_dataset: LR returned minimum --> {lr_min}")
        logger.info(f"create_meta_dataset: LR returned maximum --> {lr_max}")

        # Classify for Naive Bayes
        nb_classifier = Classifier(GaussianNB(), 'Naive Bayes')
        nb_classifier.cross_validation(data, classes, file)
        nb_sd, nb_mean, nb_max, nb_min = nb_classifier.calculate_stats()
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


# TODO: verify functionality remains the same
# TODO: remove duplicate code
def create_meta_dataset_old():
    error_count = 0
    for file in os.listdir("D:/PycharmProjects/FinalYearProject/csvDatasets"):
        dataset, data, classes = preprocess_data(path_to_csvs, file)
        rows = dataset.shape[0]  # Get rows using pandas
        columns = dataset.shape[1]  # Get columns using pandas

        # Classify for Support Vector Machine
        try:
            svm_classifier = svm.SVC()
            svm_acc = cross_validation(svm_classifier, data, classes, "SVM", file)
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
        nn_acc = cross_validation(nn_classifier, data, classes, "NN", file)
        nn_sd, nn_mean, nn_max, nn_min = calculate_stats(nn_acc)
        logger.info(f"create_meta_dataset: NN returned mean --> {nn_mean}")
        logger.info(f"create_meta_dataset: NN returned standard deviation --> {nn_sd}")
        logger.info(f"create_meta_dataset: NN returned minimum --> {nn_min}")
        logger.info(f"create_meta_dataset: NN returned maximum --> {nn_max}")

        # Classify for random forrest
        rf_classifier = RandomForestClassifier()
        rf_acc = cross_validation(rf_classifier, data, classes, "RF", file)
        rf_sd, rf_mean, rf_max, rf_min = calculate_stats(rf_acc)
        logger.info(f"create_meta_dataset: RF returned mean --> {rf_mean}")
        logger.info(f"create_meta_dataset: RF returned standard deviation --> {rf_sd}")
        logger.info(f"create_meta_dataset: RF returned minimum --> {rf_min}")
        logger.info(f"create_meta_dataset: RF returned maximum --> {rf_max}")

        # Classify for Logistic Regression
        lr_classifier = LogisticRegression()
        lr_acc = cross_validation(lr_classifier, data, classes, "LR", file)
        lr_sd, lr_mean, lr_max, lr_min = calculate_stats(lr_acc)
        logger.info(f"create_meta_dataset: LR returned mean --> {lr_mean}")
        logger.info(f"create_meta_dataset: LR returned standard deviation --> {lr_sd}")
        logger.info(f"create_meta_dataset: LR returned minimum --> {lr_min}")
        logger.info(f"create_meta_dataset: LR returned maximum --> {lr_max}")

        # Classify for Naive Bayes
        nb_classifier = GaussianNB()
        nb_acc = cross_validation(nb_classifier, data, classes, "NB", file)
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
create_meta_dataset()
# classify_metafile()


# -------------------- QUICK TESTS --------------------
# test_classifier = Classifier(svm.SVC(), 'Support Vector Machine')
# print(test_classifier)
# for accuracy in test_classifier:
#     print(accuracy)
# print('\n')
# standard_deviation, mean, maximum, minimum = test_classifier.calculate_stats()
# print(standard_deviation)
# print(mean)
# print(maximum)
# print(minimum)
# data, a, b = preprocess_data(path_to_csvs, 'updated_01a_0_100_AccGyr_0_0_0_03c_11_5f0eeba6ecd5dfaf5eb6c0c6.csv')
# test_classifier.cross_validation(a, b, 'updated_iris_basic.csv')
# print(test_classifier.accuracies)
# too_few_rows = too_few_columns = too_big = one_target_variable = fails_to_classify = 0
# too_few_rows += 1
# print(too_few_rows)
# print(too_few_columns, too_big, one_target_variable, fails_to_classify)
# verify_data_classifies(a, b

# Test new move invalid datasets
# move_invalid_datasets()

# Test cross validation and __iter__
# data, a, b = preprocess_data(path_to_csvs, 'updated_.csv')
# test_classifier.cross_validation(a, b, 'updated_.csv')
# for acc in test_classifier:
#     print(acc)
# move_invalid_datasets(f"{path_to_csvs}/updated_rbd1.csv")
# validate_datasets('D:/PycharmProjects/FinalYearProject/csvDatasets')
# for file in os.listdir("D:/PycharmProjects/FinalYearProject/csvDatasets"):
#     dataset, data, classes = preprocess_data(path_to_csvs, file)
#     rows = dataset.shape[0]  # Get rows using pandas
#     columns = dataset.shape[1]  # Get columns using pandas
#
#     # Classify for Support Vector Machine
#     try:
#         svm_classifier = Classifier(svm.SVC(), 'Support Vector Machine')
#         svm_acc = svm_classifier.cross_validation(data, classes, file)
#     except ValueError:  # If fails; skip
#         logger.error(f"{file} raised value error, skipping")
#         shutil.move(f"{path_to_csvs}/{file}", f"D:/PycharmProjects/FinalYearProject/inconsistentFailures/{file}")
#         continue
#     svm_join = ','.join(map(str, svm_acc))
#     svm_st = str(svm_acc)
#     svm_st = svm_st.replace("[", "")
#     svm_st = svm_st.replace("]", "")
#     svm_st = svm_st.replace(" ", "")
#     print(svm_st)
#     print(svm_join)
#     print(type(svm_st))
#     print(type(svm_join))
#     # write_accuracy_score(file, "SVM", svm_st, statistics.mean(svm_acc))
#     logger.info(f"Classification for {file} using classifier SVM finished!")
