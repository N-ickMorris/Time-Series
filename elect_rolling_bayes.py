# -*- coding: utf-8 -*-
"""
Trains and tests a Rolling Bayesian Ridge Regression model on data

@author: Nick
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# load dataset
Y = pd.read_csv('Y elect.csv').iloc[:,[0]]
X = pd.read_csv('X elect.csv')

# size the training and forecasting data sets
train_size = 16 # years
test_size = 1 # years

# define the initial train and test sets
train_idx = np.arange(train_size)
test_idx = np.arange(train_size, train_size + test_size)

# train the model on all the training data
pipeline = Pipeline([
    ('var', VarianceThreshold()),
    ('scale', MinMaxScaler()),
    ('model', GaussianNB()),
])
pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# produce a rolling forecast on all the testing data
predictions = pd.DataFrame()
actual = pd.DataFrame()
for i in range(X.shape[0] - train_size - test_size):
    # forecast
    pred = pd.DataFrame(pipeline.predict(X.iloc[test_idx, :])).T
    pred.columns = ["t" + str(i + 1) for i in range(pred.shape[1])]
    predictions = pd.concat([predictions, pred], axis=0).reset_index(drop=True)

    # actual
    true = Y.iloc[test_idx, :].copy().T
    true.columns = ["t" + str(i + 1) for i in range(true.shape[1])]
    actual = pd.concat([actual, true], axis=0).reset_index(drop=True)

    # roll forecast forward 
    train_idx = train_idx + 1    
    test_idx = test_idx + 1  
    
    # train a new model
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

    # report forecast
    y_pred = pred["t1"]
    y_true = Y.iloc[train_idx[-1],:][0]
    print('predicted=%f, expected=%f' % (y_pred, y_true))

# compute r2 for the predictions
acc = []
for j in predictions.columns:
    acc.append(accuracy_score(actual[j], predictions[j]))
acc
