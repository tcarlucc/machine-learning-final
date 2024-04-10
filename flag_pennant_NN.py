# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn utilities
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


# Split training and testing data
bull_flag = pd.read_csv("machine-learning-final/data/bull_flag")
bear_flag = pd.read_csv("machine-learning-final/data/bear_flag")
bull_pennant = pd.read_csv("machine-learning-final/data/bull_pennant")
bear_pennant = pd.read_csv("machine-learning-final/data/bear_pennant")

# all_data = pd.concat([bull_flag, bear_flag, bull_pennant, bear_pennant], keys=["bull_flag", "bear_flag", "bull_pennant", "bear_pennant"])
all_data = pd.concat([bull_flag, bear_flag], keys=["bull_flag", "bear_flag"])

# for value in bull_flag:
#     all_data.append([1, value])

print(all_data)



# dfX = df.iloc[:, 1:-1]
# dfY = df.iloc[:, -1:]

# # print(dfX)
# # print(dfY)

# x_train, x_test = train_test_split(dfX, test_size=.25, random_state=0)
# y_train, y_test = train_test_split(dfY, test_size=.25, random_state=0)

