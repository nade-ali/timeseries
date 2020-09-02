import pandas as pd
import time
from pandas import DataFrame
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import rc
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pylab import rcParams
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.metrics import mean_squared_error
from math import sqrt

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

df = df.replace(-99.9,np.nan)
df = df.interpolate(method = 'linear', limit_direction='both')

train_size = int(len(df) * (900/911))
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# =============================================================================
# plt.figure(figsize=(16,9), dpi=200)
# plt.plot(df['Hs'], 'b')
# plt.grid(b=None)
# plt.title('Wave Height from 1-2017 to 6-2019')
# plt.show()
# =============================================================================

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

model.add(keras.layers.Dropout(rate=0.2))

# Dense is the regular deeply connected neural network layer
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')


history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


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
