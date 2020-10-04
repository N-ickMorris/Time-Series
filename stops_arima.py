# -*- coding: utf-8 -*-
"""
Trains and tests a Holt Winter's model on data

@author: Nick
"""


import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# In[1]: Prepare the data

# read in the data
data = pd.read_csv("MplsStops.csv")

# fill in missing values
data = data.fillna(method="bfill").fillna(method="ffill")

# convert the date into a date time object
data["date"] = pd.to_datetime(data["date"])

# collect day from date
data["day"] = pd.Series(data["date"]).dt.date

# get the number of stops per day
Y = data["day"].value_counts().reset_index()
Y = Y.sort_values(by="index").drop(columns="index")
Y.columns = ["Stops"]

# split up the data into training and testing
train_fraction = 0.5 # the fraction of the data to give to training
X = Y.values
size = int(len(X) * train_fraction)
train, test = X[0:size], X[size:len(X)]


# In[2]: Train the model

# model parameters
P = 3 # number of moving average terms
D = 1 # number of times to difference
Q = 3 # number of autoregressive terms

# train and forecast one step ahead for the entire test set
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    # train
    model = ARIMA(history, order=(P, D, Q))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_fit = model.fit()

    # forecast
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

# In[2]: Visualize the predictions

# collect predictions and actual values
predictions = pd.DataFrame({"Actual": test.flatten(),
                            "Predict": np.array(predictions).flatten()}).reset_index()

# plot the prediction series
fig = px.scatter(predictions, x="index", y="Predict")
fig.add_trace(go.Scatter(x=predictions["index"], y=predictions["Actual"], mode="lines", showlegend=False, name="Actual"))
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(predictions, x="Actual", y="Predict")
fig1.add_trace(go.Scatter(x=predictions["Actual"], y=predictions["Actual"], mode="lines", showlegend=False, name="Actual"))
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")

# score the performance
print("R2: " + str(r2_score(predictions["Actual"], predictions["Predict"])))
