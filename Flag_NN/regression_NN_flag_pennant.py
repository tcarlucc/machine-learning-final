# import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# sklearn utilities
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPRegressor as nn

# Split training and testing data
bull_flag = pd.read_csv("data/bull_flag") 
bear_flag = pd.read_csv("data/bear_flag")
bull_pennant = pd.read_csv("data/bull_pennant")
bear_pennant = pd.read_csv("data/bear_pennant")

# Creates identifier column to be used to determine what type of pattern is used
bull_flag['identifier'] = 0
bear_flag['identifier'] = 1

# Combines flag data ignoring index
all_data = pd.concat([bull_flag.iloc[:,1:], bear_flag.iloc[:,1:]], ignore_index=True)

# Moves identier column to front
first_column = all_data.pop('identifier') 
all_data.insert(0, 'identifier', first_column)

X = all_data.iloc[:, : -3] # -3 ignores the conf-x and conf-y. The position of the flag shouldn't be used by NN
y = all_data.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# print(X_train)
# print(y_train)
# print(X_train.shape)
# print(ravel(y_train).shape)

regr = nn(hidden_layer_sizes=(10, 10, 10), activation='relu', solver='adam', max_iter=500, verbose=True, learning_rate='constant', learning_rate_init=.001, tol=1e-7).fit(X_train, y_train.values.ravel())
# print(regr.predict(X_test[:2]))
# print(regr.score(X_test, y_test))
print("training_score: " + str(regr.score(X_train, y_train)))
print("testing_score: " + str(regr.score(X_test, y_test)))
print("iterations: " + str(regr.n_iter_))





