# TRAINING KDD1998 CLASSIFIER

from shared_functions import *
from net_designs import *

import pandas as ps
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import keras
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


if __name__ == '__main__':
    # LOAD DATA
    print('Loading data')
    data = ps.read_csv("../kdd98_data/kdd1998tuples.csv", header=None)
    data.columns = ['customer', 'period', 'r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income',
                    'zip_region', 'zip_la', 'zip_lo', 'a', 'rew', 'r1', 'f1', 'm1', 'ir1', 'if1',
                    'gender1', 'age1', 'income1', 'zip_region1', 'zip_la1', 'zip_lo1']
    data['rew_ind'] = (data['rew'] > 0) * 1
    data['age'][data['age'] == 0] = None

    # Train and validate donation classifier
    print('Preprocessing data')

    customers = list(set(data['customer']))

    train_samples = 100000
    val_samples = 50000
    test_samples = len(customers) - val_samples - train_samples

    np.random.shuffle(customers)

    train_customers = customers[0:train_samples]
    val_customers = customers[train_samples:(train_samples + val_samples)]
    test_customers = customers[(train_samples + val_samples):]

    cols = ['r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income', 'zip_region', 'a', 'rew', 'rew_ind']

    train_data = data[data['customer'].isin(train_customers)][cols].fillna(0)
    val_data = data[data['customer'].isin(val_customers)][cols].fillna(0)
    test_data = data[data['customer'].isin(test_customers)][cols].fillna(0)

    n_train = train_data.shape[0]
    n_val = val_data.shape[0]
    n_test = test_data.shape[0]

    cols_X = ['r0', 'f0', 'm0', 'ir0', 'if0', 'gender', 'age', 'income', 'zip_region', 'a']
    cols_Y = ['rew_ind']

    x_train = train_data[cols_X].values.astype(np.float32)
    y_train = train_data[cols_Y].values.astype(np.int32)

    x_val = val_data[cols_X].values.astype(np.float32)
    y_val = val_data[cols_Y].values.astype(np.int32)

    x_test = test_data[cols_X].values.astype(np.float32)
    y_test = test_data[cols_Y].values.astype(np.int32)

    file_name = "../results/kdd98_propagation_classifier_best.h5"

    model = KDDClassifier()
    model.build(input_shape=[100, 10])
    model.load_weights(file_name)
    y_score = model.predict(x_test)
    roc(to_categorical(y_test), y_score, name="../results/kdd98_propagation_classifier_roc5.pdf")