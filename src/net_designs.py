# NET DESIGNS

import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32"

import numpy as np
import random

from matplotlib import rcParams
rcParams['font.family'] = 'Euclid'
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

# MODELS

def KDDClassifier():

    model = Sequential()

    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    return model


def KDDRegressor():

    model = Sequential()

    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, input_dim=100))

    return model


def VSRegressor():

    model = Sequential()

    model.add(Dense(100, input_dim=20*8))
    model.add(Activation("relu"))
    model.add(Dense(100, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(100, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(100, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(100, input_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(20, input_dim=100))

    return model
