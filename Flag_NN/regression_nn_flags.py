# import needed libraries
from __future__ import absolute_import, division, print_function, unicode_literals

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers, losses

from sklearn.model_selection import train_test_split


def z_score_normalization(data):
    for column in data.columns:
        data[column] = (data[column] - np.mean(data[column])) / np.std(data[column])

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [price]')
  plt.legend()
  plt.grid(True)
  plt.show()

def get_and_preprocess_data():
    # Split training and testing data
    bull_flag = pd.read_csv("data/bull_flag") 
    bear_flag = pd.read_csv("data/bear_flag")

    # Creates identifier column to be used to determine what type of pattern is used
    bull_flag['identifier'] = 0
    bear_flag['identifier'] = 1

    # Combines flag data ignoring index
    all_data = pd.concat([bull_flag.iloc[:,1:], bear_flag.iloc[:,1:]], ignore_index=True)

    # Moves identier column to front
    first_column = all_data.pop('identifier') 
    all_data.insert(0, 'identifier', first_column)

    z_score_normalization(all_data)

    X = all_data.iloc[:, : -3] # -3 ignores the conf-x and conf-y. The position of the flag shouldn't be used by NN
    y = all_data.iloc[:, -1:]

    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

def neural_network():
    layers = [
    Flatten(),
    # Add hidden layers
    Dense(100, activation = 'relu'), 
    Dense(100, activation = 'relu'), 
    Dense(100, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(100, activation = 'relu'),
    Dense(100, activation = 'relu'),

    #Add output layer
    Dense(1)
    ]

    model = Sequential(layers)

    model.compile(optimizer=optimizers.Adam(learning_rate=.01), loss=losses.MeanAbsoluteError())

    return model




def main():
    get_and_preprocess_data()
    
    model = neural_network()
    
    history = model.fit(X_train, y_train, validation_split=.2, epochs=100 , validation_data=(X_test, y_test))
    
    # plot_loss(history)

    test_predictions = model.predict(X_test).flatten().reshape(29,1)

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, test_predictions)
    # plt.xlabel('True Values [retun]')
    # plt.ylabel('Predictions [return]')

    error = test_predictions - y_test
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [return]')
    _ = plt.ylabel('Count')
    plt.show()





main()
