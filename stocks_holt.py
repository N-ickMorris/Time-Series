# -*- coding: utf-8 -*-
"""
Trains and tests a Holt-Winter's model on data

@author: Nick
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import Holt
from sklearn.metrics import r2_score
from plots import parity_plot, series_plot


# In[1]: Train the model

# load dataset
Y = pd.read_csv('Y stocks.csv').iloc[:,[2]]

# split up the data into training and testing
X = Y.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]

# train and forecast one step ahead for the entire test set
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = Holt(history)
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

# collect predictions and actual values
predictions = pd.DataFrame({"Actual": test.flatten(), 
                            "Predict": np.array(predictions).flatten()})
# predictions.to_csv("holt predictions.csv", index=False)

# In[2]: Visualize the predictions

save_plot = False

# series plot
series_plot(predict=predictions["Predict"], actual=predictions["Actual"], 
            title="Holt-Winter's One-Step Ahead Rolling Forecast - Test",
            save=save_plot)

# parity plot
r2 = str(np.round(r2_score(predictions["Actual"], predictions["Predict"]) * 100, 1)) + "%"
parity_plot(predict=predictions["Predict"], actual=predictions["Actual"],
            title="Holt-Winter's Parity Plot - Test - R2: " + r2, save=save_plot)
