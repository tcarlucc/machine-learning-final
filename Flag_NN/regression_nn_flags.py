# import needed libraries
from __future__ import absolute_import, division, print_function, unicode_literals

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import statistics as stat


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import datetime
import os

# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers as optimizer
from tensorflow.keras import losses as loss
from tensorflow.keras import activations as activation
from tensorflow.keras import layers as layers
from tensorflow.keras import metrics as metric

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
        # add else to normalize return too!                 
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

def neural_network(nn_activation, nn_optimizer):
    model = Sequential()

    model.add(Flatten())
    model.add(Dense(100, activation = nn_activation))
    model.add(Dense(100, activation = nn_activation))
    model.add(Dense(100, activation = nn_activation))
    model.add(Dense(1, activation='linear'))


    model.compile(optimizer=nn_optimizer, loss=loss.MeanSquaredError(), metrics=[metric.MeanAbsoluteError()])

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
    for i in range(5):
    
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

def get_opt_name(opt):
    """
    Creates a file name showing the optimizer and activation function to be used with tensorflow logs
    """
    to_return = ""
    match opt:
        case optimizer.legacy.Adagrad():
            to_return += "Adagrad_"
        case optimizer.legacy.Adam():
            to_return += "Adam_"
        case optimizer.legacy.RMSprop():
            to_return += "RMSProp_"
        case optimizer.legacy.SGD():
            to_return += "SGD_"
        case _:
            to_return += "error_"
        
    return to_return


def run_sample_nn_structure():
    bull_path = "data/new_data/bull_flag"
    bear_path = "data/new_data/bear_flag"
    test_size = .35
    MAX_EPOCHS = 30

    X_train, X_val, y_train, y_val = get_and_preprocess_data(bull_path, bear_path, test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size)
    
    optimizers = [optimizer.legacy.Adagrad(), optimizer.legacy.Adam(), optimizer.legacy.RMSprop(), optimizer.legacy.SGD()]
    activations = ["tanh", "sigmoid", "relu", "leaky_relu"]

    os.system("rm -rf ./logs/")

    data = pd.DataFrame(columns=["optimizer", "activation", "test_avg", "test_best", "train_avg", "train_best", "run1", "run2", "run3"])

    for opt in optimizers:
        for act in activations:
            test_maes = []
            train_maes = []
            for i in range(3):
                model = neural_network(act, opt)
                label = get_opt_name(opt)
                log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + label + act
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                
                history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, verbose=0, validation_data=(X_val, y_val), callbacks=[tensorboard_callback])

                evaluation = model.evaluate(X_test, y_test, batch_size=16, verbose=0)

                test_maes.append(evaluation[1])
                train_maes.append(min(history.history['mean_absolute_error']))

            toAdd = {'optimizer': get_opt_name(opt), 'activation': act, "test_avg":stat.mean(test_maes), 
                     "test_best": min(test_maes), "train_avg":stat.mean(train_maes), "train_best": min(train_maes),
                       'run1': test_maes[0], 'run2': test_maes[1], 'run3': test_maes[2]}
            toAdd = pd.DataFrame(toAdd, index=[0])
            display(toAdd)
            data = pd.concat([data, toAdd], ignore_index = True)

    display(data)

    print("return_std: " + str(return_std))
    print("return_mean: " + str(return_mean))

    os.system("tensorboard --logdir logs/fit")

def main():
    # run_with_single_return()
    # run_single_return_early_stopping(75)
    run_sample_nn_structure()


main()
