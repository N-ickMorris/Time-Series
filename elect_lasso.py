# -*- coding: utf-8 -*-
"""
Trains and tests a Lasso regression model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from plots import matrix_plot, parity_plot, series_plot

TIME_SERIES = True

# In[1]: Train the model

# read in the data
X = pd.read_csv("X elect.csv")
Y = pd.read_csv("Y elect.csv")

# standardize the inputs to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])
outputs = Y.shape[1]

# separate the data into training and testing
if TIME_SERIES:
    test_idx = X.index.values[-int(X.shape[0] / 12):]
else:
    np.random.seed(1)
    test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 12), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# set up the model
if classifier:
    model = MultiOutputClassifier(LogisticRegressionCV(penalty="l1", solver="saga",
                                                       Cs=20, cv=3, tol=1e-4,
                                                       max_iter=500,
                                                       class_weight="balanced",
                                                       random_state=42,
                                                       n_jobs=1))
else:
    model = MultiOutputRegressor(LassoCV(eps=1e-9, n_alphas=20, cv=3,
                                         tol=1e-4, max_iter=500, random_state=42,
                                         n_jobs=1))

# train the model
model.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# In[2]: Collect the predictions

# predict training and testing data
train_predict = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=Y.columns)

# reshape all of the predictions into a single table
predictions = pd.DataFrame()
for j in range(outputs):
    # collect training data
    predict_j = np.array(train_predict.iloc[:,j])
    actual_j = np.array(Y.iloc[train_idx, j])
    name_j = Y.columns[j]
    data_j = "Train"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            len(train_idx)),
                                          "Data": np.repeat(data_j,
                                                            len(train_idx))})],
                            axis="index")

    # collect testing data
    predict_j = np.array(test_predict.iloc[:,j])
    actual_j = np.array(Y.iloc[test_idx, j])
    name_j = Y.columns[j]
    data_j = "Test"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            len(test_idx)),
                                          "Data": np.repeat(data_j,
                                                            len(test_idx))})],
                            axis="index")
predictions = predictions.reset_index(drop=True)
# predictions.to_csv("lasso predictions.csv", index=False)

# In[3]: Visualize the predictions

# confusion matrix heatmap
save_plot = False
if classifier:
    for j in Y.columns:
        # training data
        data_j = "Train"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        accuracy = str(np.round(accuracy_score(data["Actual"],
                                               data["Predict"]) * 100, 1)) + "%"
        matrix = confusion_matrix(data["Actual"], data["Predict"])
        matrix_plot(matrix, title=j + " - " + data_j + " - Accuracy: " + accuracy,
                    save=save_plot)

        # testing data
        data_j = "Test"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        accuracy = str(np.round(accuracy_score(data["Actual"],
                                               data["Predict"]) * 100, 1)) + "%"
        matrix = confusion_matrix(data["Actual"], data["Predict"])
        matrix_plot(matrix, title=j + " - " + data_j + " - Accuracy: " + accuracy,
                    save=save_plot)

# parity plot
else:
    for j in Y.columns:
        # training data
        data_j = "Train"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        r2 = str(np.round(r2_score(data["Actual"], data["Predict"]) * 100, 1)) + "%"
        parity_plot(predict=data["Predict"], actual=data["Actual"],
                    title=j + " - " + data_j + " - R2: " + r2, save=save_plot)

        # testing data
        data_j = "Test"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        r2 = str(np.round(r2_score(data["Actual"], data["Predict"]) * 100, 1)) + "%"
        parity_plot(predict=data["Predict"], actual=data["Actual"],
                    title=j + " - " + data_j + " - R2: " + r2, save=save_plot)

        if TIME_SERIES:
            # series plot
            series_plot(predict=data["Predict"], actual=data["Actual"], 
                        title=j + " - " + data_j + " - Forecast: ", save=save_plot)

# In[4]: Predict the 2020 Election

# collect the column names (X elect.csv)
columns_2020 = ['RepublicanFraction Ohio(t-2)', 
                'RepublicanFraction Alaska RepublicanFraction New Hampshire(t-2)', 
                'RepublicanFraction Arizona RepublicanFraction Idaho(t-2)', 
                'RepublicanFraction Arizona RepublicanFraction North Dakota(t-2)', 
                'RepublicanFraction Washington RepublicanFraction Wyoming(t-2)', 
                'RepublicanFraction Wisconsin RepublicanFraction Wyoming(t-2)', 
                'RepublicanFraction Ohio(t-1)', 
                'RepublicanFraction Alaska RepublicanFraction New Hampshire(t-1)', 
                'RepublicanFraction Arizona RepublicanFraction Idaho(t-1)', 
                'RepublicanFraction Arizona RepublicanFraction North Dakota(t-1)', 
                'RepublicanFraction Washington RepublicanFraction Wyoming(t-1)', 
                'RepublicanFraction Wisconsin RepublicanFraction Wyoming(t-1)', 
                ' DemocratWon(t-3)', 
                ' DemocratWon(t-2)', 
                ' DemocratWon(t-1)']

# collect the data (X elect features.csv, Y elect features.csv)
data_2020 = [[
0.484859432307505, 
0.270334720629146, 
0.362749645467386, 
0.328287627398856, 
0.301519715920963, 
0.331157182714036,
0.542676845771359, 
0.29077791009483, 
0.354461666649614, 
0.362240324750926, 
0.31200485463974, 
0.381615292198692,
1,
1,
0
]]

# predict the election
df_2020 = pd.DataFrame(data=data_2020, columns=columns_2020, index=["2020"])
predict_2020 = pd.DataFrame(model.predict(df_2020), columns=Y.columns, index=df_2020.index)
print(predict_2020)
