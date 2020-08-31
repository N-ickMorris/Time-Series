# -*- coding: utf-8 -*-
"""
Creates lagged features for time series

@author: Nick
"""


import pandas as pd


# convert series to supervised learning
def series_to_supervised(data, n_in_low=0, n_in_up=1, n_out_low=0, n_out_up=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in_up, n_in_low, -1):
		cols.append(df.shift(i))
		names += [(str(df.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(n_out_low, n_out_up):
		cols.append(df.shift(-i))
		if i == 0:
			names += [str(df.columns[j]) + '(t)' for j in range(n_vars)]
		else:
			names += [(str(df.columns[j]) + '(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# read in the data
X = pd.read_csv("X elect.csv")
Y = pd.read_csv("Y elect.csv")

# save a copy
X.to_csv("X elect features.csv", index=False)
Y.to_csv("Y elect features.csv", index=False)

# shift Y by LAGS
outputs = Y.shape[1]
Y = series_to_supervised(Y, n_in_low=0,
                         n_in_up=3, 
                         n_out_low=0,
                         n_out_up=1)

# shift X by LAGS
X = series_to_supervised(X, n_in_low=0,
                         n_in_up=2, 
                         n_out_low=0,
                         n_out_up=0)

# add lags to features and remove the first LAGS rows
X = pd.concat([X, Y.iloc[:,:-outputs]], axis=1).reset_index(drop=True)
Y = Y.iloc[:,-outputs:].reset_index(drop=True)

# drop rows with missing values
X = X.dropna(axis=0).reset_index(drop=True)

# export the data
X.to_csv("X elect.csv", index=False)
Y.to_csv("Y elect.csv", index=False)
