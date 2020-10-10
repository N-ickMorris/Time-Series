# -*- coding: utf-8 -*-
"""
Modeling Bike Share Demand with Lasso Regression

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# In[1]: Prepare the data

# read in the data
train = pd.read_csv("bike_train.csv")
test = pd.read_csv("bike_test.csv")

# define the input variables
inputs = test.columns.tolist()

# define training and testing indices
train_idx = np.arange(train.shape[0])
test_idx = np.arange(test.shape[0]) + train.shape[0]

# split up the training into set A and set B for scoring
size = int(train.shape[0] / 2)
trainA_idx = train_idx[:size]
trainB_idx = train_idx[size:]

# split up the data into inputs (X) and outputs (Y)
X = pd.concat([train[inputs], test], axis=0).reset_index(drop=True)
Y = train[["count"]]

# convert datetime to a date time object
X["datetime"] = pd.to_datetime(X["datetime"])

# collect the hour
hours = pd.get_dummies(pd.Series(X["datetime"]).dt.hour.astype(str))
hours.columns = ["Hour " + str(c) for c in hours.columns]

# collect the weekday
weekday = pd.get_dummies(pd.Series(X["datetime"]).dt.weekday.astype(str))
weekday.columns = ["Weekday " + str(c) for c in weekday.columns]

# collect the week
week = pd.get_dummies(pd.Series(X["datetime"]).dt.week.astype(str))
week.columns = ["Week " + str(c) for c in week.columns]

# collect the month
month = pd.get_dummies(pd.Series(X["datetime"]).dt.month.astype(str))
month.columns = ["Month " + str(c) for c in month.columns]

# add the features to the data
X = pd.concat([X, hours, weekday, week, month], axis=1)

# make datetime the index
X.index = X["datetime"]
X = X.drop(columns="datetime")

# convert season and weather to string variables
X["season"] = X["season"].astype(str)
X["weather"] = X["weather"].astype(str)

# determine which columns are strings (for X)
x_columns = X.columns
x_dtypes = X.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
X = pd.get_dummies(X, columns=x_columns[x_str])

# In[1]: Model the data

# set up cross validation for time series
tscv = TimeSeriesSplit(n_splits=5)
folds = tscv.get_n_splits(X)

# set up a machine learning pipeline
pipeline = Pipeline([
    ('var1', VarianceThreshold()),
    ('poly', PolynomialFeatures(2)),
    ('var2', VarianceThreshold()),
    ('scale', MinMaxScaler()),
    ('model', LassoCV(cv=folds, eps=1e-9, n_alphas=16, n_jobs=-1)),
])

# train a model
pipeline.fit(X.iloc[trainA_idx, :], Y.iloc[trainA_idx, :])

# forecast
predict = pipeline.predict(X.iloc[trainB_idx, :])
actual = Y.iloc[trainB_idx, :].to_numpy().T[0]

# score the forecast
print("R2: " + str(r2_score(actual, predict)))

# prepare the data for plotting
df = pd.DataFrame({"Predict": predict, "Actual": actual})
df["index"] = X.iloc[trainB_idx, :].index

# plot the prediction series
fig = px.line(df, x="index", y="Predict")
fig.add_trace(go.Scatter(x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(go.Scatter(x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")


