# TRAINING KDD1998 CLASSIFIER

from src.shared_functions import *
from src.net_designs import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import torch.optim as optim
import os

from copy import deepcopy

import pandas as ps
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc


def Train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 4000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def Validate(model, val_loader, best_model, best_acc):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)

    if accuracy > best_acc:
        best_model = model
        best_acc = accuracy
        torch.save(model.state_dict(), "/bigdisk/lax/renaza/env/CustomerSim/models/kdd_classifier.pt")

    return best_model, best_acc


def Test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_score = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            y_score.append(output)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    y_score = torch.exp(torch.cat(y_score, 0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return y_score


if __name__ == '__main__':

    # seed

    RANDOM_SEED = 777
    n_epochs = 50
    batch_size = 100
    device = "cuda:0"

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    # cupy.random.seed(RANDOM_SEED)

    # LOAD DATA
    print('Loading data')
    data = ps.read_csv("/bigdisk/lax/renaza/env/CustomerSim/kdd98_data/kdd1998tuples.csv", header=None)
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

    x_train = torch.tensor(train_data[cols_X].values.astype(np.float32), device=device)
    y_train = torch.tensor(train_data[cols_Y].values.astype(np.int64), device=device).squeeze()

    x_val = torch.tensor(val_data[cols_X].values.astype(np.float32), device=device)
    y_val = torch.tensor(val_data[cols_Y].values.astype(np.int64), device=device).squeeze()

    x_test = torch.tensor(test_data[cols_X].values.astype(np.float32), device=device)
    y_test = torch.tensor(test_data[cols_Y].values.astype(np.int64), device=device).squeeze()

    dataset_train = TensorDataset(x_train, y_train)
    dataset_val = TensorDataset(x_val, y_val)
    dataset_test = TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size)

    # DEFINE NEURAL NET
    print('Training KDD98 neural net classifier')

    # Define the kdd98 classifier model
    model = KDDClassifier().to(device)
    best_model = model
    optimizer = optim.Adam(model.parameters())
    best_val_acc = 0
    y_score = Test(model, test_loader)

    for epoch in range(1, n_epochs + 1):
        Train(model, train_loader, optimizer, epoch)
        best_model, best_val_acc = Validate(model, val_loader, best_model, best_val_acc)
        y_score = Test(model, test_loader)

    print("Final Results: ")
    y_score = Test(best_model, test_loader)

    # model.load_weights(file_name)

    # VALIDATE CLASSIFIER
    print('Validating neural net classifier')

    roc(F.one_hot(y_test).cpu().numpy(), y_score.cpu().numpy(),
        name="/bigdisk/lax/renaza/env/CustomerSim/results/kdd98_propagation_classifier_roc.pdf")

    """
    # TRAIN RANDOM FOREST
    print('Training random forest classifier')
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train.ravel())

    # VALIDATE RANDOM FOREST
    print('Validating random forest classifier')
    y_score = clf.predict_proba(x_test)

    # SAVE ROC CURVE PLOT
    roc(to_categorical(y_test), y_score, name="../results/kdd98_propagation_classifier_roc_rf.pdf")
    """
