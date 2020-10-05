import pandas as pd
from pandas import DataFrame
from pandas.plotting import autocorrelation_plot
from pylab import rcParams
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf

from math import sqrt
import time

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import pyplot

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow import keras



sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 22, 10
mpl.rcParams['axes.grid'] = False

# =============================================================================
# # ------------------------ DATA IMPORT ------------------------
# =============================================================================
df = pd.read_csv(
  "waves.csv",
  parse_dates=['Date/Time']
)

# Missing values in the data are represented with -99.9
# Once the missing values are removed, we need to interpolate to fill gaps
# Finally, split into training and test sets
df = df.replace(-99.9,np.nan)
df = df.interpolate(method = 'linear', limit_direction='both')

train_size = int(len(df) * (900/911))
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]


# We need to normalize the x variables
# Robustscaler essentially normalizes the data
# The way to use it is to first create a "transformer" variable
# The variable is "fitted", i.e. the median and quartiles are calculated
# Replace the original train & test data with the "transformed" data
# The "transformed" data has been centred and scaled.
f_columns = ['Hs']
f_transformer = RobustScaler()
f_transformer = f_transformer.fit(train[f_columns].to_numpy())
train.loc[:, f_columns] = f_transformer.transform(
  train[f_columns].to_numpy()
)
test.loc[:, f_columns] = f_transformer.transform(
  test[f_columns].to_numpy()
)

# Normalize the count (y-variable) as well
cnt_transformer = RobustScaler()
cnt_transformer = cnt_transformer.fit(train[['Hs']])
train.loc[:,'Hs'] = cnt_transformer.transform(train[['Hs']])
test.loc[:,'Hs'] = cnt_transformer.transform(test[['Hs']])

train = train[['Hs']]
test = test[['Hs']]

# Create a function to generate a time series dataset
# the function works like this: first a "history size" is specified
# The "history size" refers to how long a single "time string" should be
# The number of strings created is len(training set) - history size
# For the y-var, the first "history size" number of values are removed
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_train, y_train = create_dataset(train, train.Hs, time_steps)
X_test, y_test = create_dataset(test, test.Hs, time_steps)

# =============================================================================
# # ------------------------ MODEL CREATION ------------------------
# =============================================================================

# Sequential refers to a model in which property layers are added
# The first layer that we add is LSTM within the bidirectional layer

# What are units?
"""
The proper intuitive explanation of the 'units' parameter for 
Keras recurrent neural networks is that with units = 1 you get a RNN 
as described in textbooks, and with units = n you get a layer which 
consists of n independent copies of such RNN 
- they'll have identical structure, but as they'll 
be initialized with different weights
"""

# What is the input_shape?
"""
A time series input is (typically) a 3D input which has the 
following dimensions:

Batch dimension - number of samples/rows in a batch
Time dimension - represents the temporal aspect of your data 
(e.g. number of days). Here it is "time_steps".
Input dimension - number of features in a single input 
and a single timestep or basically, the number of variables.
"""
t = time.time()
model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)

# The Dropout layer randomly sets input units to 0 with a frequency 
# of rate at each step during training time, which helps 
# prevent overfitting.
model.add(keras.layers.Dropout(rate=0.2))

# Dense is the regular deeply connected neural network layer
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Epochs refer to how many times the entire training set 
# is to be run through
# Batch size refers to how the training set is to be broken down
# If there are a total of 320 rows in the training data, then 
# The fitting runs 10 times per epoch. 
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

# val_loss is the value of cost function for the 
# cross-validation data and loss is the value of 
# cost function for the training data.
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Use the model to predict y for the test data
y_pred = model.predict(X_test)

# Inverse transform takes the data back to its original form
y_train_inv = f_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = f_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = f_transformer.inverse_transform(y_pred)

print("Text Preprocessing complete.")
print(f"Time Taken: {round(time.time()-t)} seconds")

# Same thing as above but just don't show training data
plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Bike Count')
plt.xlabel('Time Step')
plt.legend()
plt.show();

# Determine parameters for the ARIMA model
autocorrelation_plot(train)
pyplot.show()

result=adfuller(train.diff().dropna())
print(result[1])

plot_pacf(train.diff().dropna(), lags = 10)
plot_acf(train.diff().dropna(), lags = 10)

t = time.time()
model1 = ARIMA(train, order=(1,1,1))
model1_fit = model1.fit()
print(model1_fit.summary())
# plot residual errors
residuals = DataFrame(model1_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


# Build Model
history=[x for x in train.values]    
predictions = list()
for t in range(len(test)):
	model3 = ARIMA(history, order=(1,1,1))
	model3_fit = model3.fit()
	output = model3_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test.values[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
y_arima_inv = f_transformer.inverse_transform(np.reshape(predictions,(-1,1)))
print("Text Preprocessing complete.")
print(f"Time Taken: {round(time.time()-t)} seconds")

plt.figure(figsize=(22,10), dpi=100)
plt.plot(y_test_inv.flatten(), label='actual')
plt.plot(y_arima_inv[10:528], label='ARIMA')
plt.plot(y_pred_inv.flatten(), 'r', label="LSTM")
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

rmse = sqrt(mean_squared_error(y_test_inv.flatten(),y_arima_inv[10:528]))
print('Test RMSE: %.3f' % rmse)

rmse = sqrt(mean_squared_error(y_test_inv.flatten(),y_pred_inv.flatten()))
print('Test RMSE: %.3f' % rmse)
