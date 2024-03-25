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
df = pd.read_csv("data/bull_flag")

dfX = df.iloc[:, 1:-1]
dfY = df.iloc[:, -1:]

# print(dfX)
# print(dfY)

x_train, x_test = train_test_split(dfX, test_size=.25, random_state=0)
y_train, y_test = train_test_split(dfY, test_size=.25, random_state=0)

