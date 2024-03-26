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

def conf_matrix_to_df(conf_matrix):
    return pd.DataFrame(conf_matrix)


# Split training and testing data
df = pd.read_csv("data/bull_flag")

dfX = df.iloc[:, 1:-1]
dfY = df.iloc[:, -1:]

for i, row in dfY.iterrows():
    # print(row[0])
    val = 0
    if row[0] > 0:
        val = 1
    row[0] = val


# print(dfX)
print(dfY)

x_train, x_test = train_test_split(dfX, test_size=.25, random_state=.4)
y_train, y_test = train_test_split(dfY, test_size=.25, random_state=.4)

model = SVC(kernel="linear")
model.fit(x_train, y_train)
results = model.predict(x_test)
print(results)

print(conf_matrix_to_df(confusion_matrix(y_test, results)))
