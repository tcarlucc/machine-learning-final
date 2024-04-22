# import needed libraries
from __future__ import absolute_import, division, print_function, unicode_literals

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import datetime
import os

# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers, losses

from sklearn.model_selection import train_test_split

import pathlib
import shutil
import tempfile


def z_score_normalization(data):
    for column in data.columns:
        if (column == 'return_1.0'):
            global return_std, return_mean
            return_mean = np.mean(data[column])
            return_std = np.std(data[column])
        data[column] = (data[column] - np.mean(data[column])) / np.std(data[column])

def unnormalize_return(value, data):
    return (value * return_std) + return_mean

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error [price]')
  plt.legend()
  plt.grid(True)
  plt.show()

def get_and_preprocess_data(bull_path, bear_path, test_size):
    # Split training and testing data
    bull_flag = pd.read_csv(bull_path) 
    bear_flag = pd.read_csv(bear_path)

    # Creates identifier column to be used to determine what type of pattern is used
    bull_flag['identifier'] = 0
    bear_flag['identifier'] = 1

    # Combines flag data ignoring index
    global all_data
    all_data = pd.concat([bull_flag.iloc[:,1:], bear_flag.iloc[:,1:]], ignore_index=True)

    # Moves identier column to front
    first_column = all_data.pop('identifier') 
    all_data.insert(0, 'identifier', first_column)

    z_score_normalization(all_data)
    X = all_data.iloc[:, :-1] # -3 ignores the conf-x and conf-y. The position of the flag shouldn't be used by NN
    y = all_data.iloc[:, -1:]

    global X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=test_size)


def neural_network(batch_size, steps_per_epoch, n_hidden_layers):
    model = Sequential()

    model.add(Flatten())
    for i in range(n_hidden_layers + 1):
        model.add(Dense(100, activation = 'relu'))
    model.add(Dense(1, activation='linear'))

    # Decaying learning rate is helpful
    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    #     0.001,
    #     decay_steps=steps_per_epoch * 1000,
    #     decay_rate=1,
    #     staircase=False)

    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError(), metrics=[keras.metrics.MeanAbsoluteError()])

    return model

def run_with_single_return():
    bull_path = "data/bull_flag"
    bear_path = "data/bear_flag"
    test_size = .2

    X_train, X_test, y_train, y_test = get_and_preprocess_data(bull_path, bear_path, test_size)

    BATCH_SIZE = 32
    STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
    
    model = neural_network(BATCH_SIZE, STEPS_PER_EPOCH, 3)

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size)

    os.system("rm -rf ./logs/")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
    history = model.fit(X_train, y_train, epochs=30, verbose=0, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

    print("Average test loss: ", np.average(history.history['mean_absolute_error']))
    print("Average test loss converted: ", unnormalize_return(np.average(history.history['mean_absolute_error']), all_data))
    print(history)

    os.system("tensorboard --logdir logs/fit")

def run_single_return_early_stopping(max_epochs):
    bull_path = "data/new_data/bull_flag"
    bear_path = "data/new_data/bear_flag"
    test_size = .2

    X_train, X_test, y_train, y_test = get_and_preprocess_data(bull_path, bear_path, test_size)

    BATCH_SIZE = 16
    STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
    
    

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size)
    os.system("rm -rf ./logs/")
    for i in range(20):
    
        model = neural_network(BATCH_SIZE, STEPS_PER_EPOCH, (i%5) + 1)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        
        history = model.fit(X_train, y_train, epochs=max_epochs, verbose=0, validation_data=(X_test, y_test), callbacks=[tensorboard_callback, early_stopping])
        print("model " + str(i) + " done!")

    # print("Average test loss: ", np.average(history.history['mean_absolute_error']))
    # print("Min test loss: ", np.min(history.history['mean_absolute_error']))
    # print("Min test loss converted: ", unnormalize_return(np.min(history.history['mean_absolute_error']), all_data))

    os.system("tensorboard --logdir logs/fit")
    
    # plot_loss(history)


    # print("Evaluate on test data")
    # results = model.evaluate(X_test, y_test, batch_size=128)
    # print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions for 3 samples")
    # predictions = model.predict(X_test[:3])
    # print("predictions shape:", predictions.shape)
    
    # print(X_test[:3])
    # print(predictions)

    # test_predictions = model.predict(X_test).flatten().reshape(29,1)

    # a = plt.axes(aspect='equal')
    # plt.scatter(y_test, test_predictions)
    # plt.xlabel('True Values [return]')
    # plt.ylabel('Predictions [return]')

    # error = test_predictions - y_test
    # plt.hist(error, bins=25)
    # plt.xlabel('Prediction Error [return]')
    # _ = plt.ylabel('Count')
    # plt.show()


def main():
    # run_with_single_return()
    run_single_return_early_stopping(75)


main()
