import logging
import sys
import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

path_to_csvs = "D:/PycharmProjects/FinalYearProject/csvDatasets"

# an array where the accuracy will be grouped in batches of 10% i.e. index 0 = 0 ≤ x ≤ 10, index 1 = 10 <  x ≤ 20, etc
accuracy_array = [0] * 10
strs = [""] * 10
csv_files = os.listdir(path_to_csvs)
print(accuracy_array)
print(strs)
failed_files = []

for file in csv_files:
    logger.info(f"File to be classified --> {file}")
    data = pd.read_csv(f"{path_to_csvs}/{file}", header=None)

    # Get number of columns then remove 1 to fit array format for later processing
    x = len(data.columns)
    if x <= 2:
        logger.warning(f"File has too few columns, skipping file --> {file}")
        failed_files.append(file)
        continue

    # If greater than 2MB
    if os.path.getsize(f"{path_to_csvs}/{file}") > 209715200:
        logger.warning(f"File is too big, skipping file --> {file}")
        continue
    x -= 1

    # All columns except the last one
    a = data.drop(data.columns[len(data.columns) - 1], axis=1)
    # Only the last column
    b = data.drop(data.iloc[:, 0:x], axis=1)

    # Run classifier after data processing
    classifier = tree.DecisionTreeClassifier(criterion='gini')
    try:
        classifier.fit(a, b)
        # Evaluate model on testing data
        y_pred = classifier.predict(a)
        acc = accuracy_score(y_pred, b)
        logger.info(f"Accuracy score for file {file} returned --> {acc}")
    except:
        logger.warning(f"Something went wrong with file --> {file}")
        failed_files.append(file)

    # Collect Statistics on classified data
    if acc <= .10:
        temp = accuracy_array[0]
        temp += 1
        accuracy_array.insert(0, temp)
        stemp = strs[0]
        stemp = stemp + f" {file}"
        strs.insert(0, stemp)
    elif .10 < acc <= .20:
        temp = accuracy_array[1]
        temp += 1
        accuracy_array.insert(1, temp)
        stemp = strs[1]
        stemp = stemp + f" {file}"
        strs.insert(1, stemp)
    elif .20 < acc <= .30:
        temp = accuracy_array[2]
        temp += 1
        accuracy_array.insert(2, temp)
        stemp = strs[2]
        stemp = stemp + f" {file}"
        strs.insert(2, stemp)
    elif .30 < acc <= .40:
        temp = accuracy_array[3]
        temp += 1
        accuracy_array.insert(3, temp)
        stemp = strs[3]
        stemp = stemp + f" {file}"
        strs.insert(3, stemp)
    elif .40 < acc <= .50:
        temp = accuracy_array[4]
        temp += 1
        accuracy_array.insert(4, temp)
        stemp = strs[4]
        stemp = stemp + f" {file}"
        strs.insert(4, stemp)
    elif .50 < acc <= .60:
        temp = accuracy_array[5]
        temp += 1
        accuracy_array.insert(5, temp)
        stemp = strs[5]
        stemp = stemp + f" {file}"
        strs.insert(5, stemp)
    elif .60 < acc <= .70:
        temp = accuracy_array[6]
        temp += 1
        accuracy_array.insert(6, temp)
        stemp = strs[6]
        stemp = stemp + f" {file}"
        strs.insert(6, stemp)
    elif .70 < acc <= .80:
        temp = accuracy_array[7]
        temp += 1
        accuracy_array.insert(7, temp)
        stemp = strs[7]
        stemp = stemp + f" {file}"
        strs.insert(7, stemp)
    elif .80 < acc <= .90:
        temp = accuracy_array[8]
        temp += 1
        accuracy_array.insert(8, temp)
        stemp = strs[8]
        stemp = stemp + f" {file}"
        strs.insert(8, stemp)
    else:
        temp = accuracy_array[9]
        temp += 1
        accuracy_array.insert(9, temp)
        stemp = strs[9]
        stemp = stemp + f" {file}"
        strs.insert(9, stemp)

print("Accuracy score distribution:")
print(accuracy_array)
print("Corresponding files which scored in bands of 10%:")
print(strs)
print("Files which failed to classify:")
print(failed_files)
print(f"Number of failed files --> {len(failed_files)}")
# min_score = 0
# max_score = .1
# for i in accuracy_array:
#     logger.info(f"These files had a score of between {min_score} and {max_score}")
#     print(strs[i])
#     logger.info(f"These are their scores in order:")
#     print(accuracy_array[i])
#     min_score += .1
#     max_score += .1
