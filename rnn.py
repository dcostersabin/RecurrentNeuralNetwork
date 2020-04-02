import numpy as np
import matplotlib as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# importing dataset from the directory
BASE = os.getcwd()
TRAINING_DIR = BASE + '/Dataset/Training/Google_Stock_Price_Train.csv'
TEST_DIR = BASE + '/Dataset/Testing/Google_Stock_Price_Test.csv'
MODEL_DIR = BASE + '/Model/rnnModel'
# retrieving the data from csv file
dataset_train = pd.read_csv(TRAINING_DIR)
# getting the feature column
training_set = dataset_train.iloc[:, 1:2].values
# initializing the scale
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)
# structuring the data
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# reshaping the dataset
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# initializing the rnn
regressor = Sequential()
# adding the first long short term memory
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
# adding the dropout layer to overcome over fitting
regressor.add(Dropout(rate=0.2))
# adding another long short term memory
regressor.add(LSTM(units=50, return_sequences=True))
# adding another  dropout layer
regressor.add(Dropout(rate=0.2))
# adding another long short term memory
regressor.add(LSTM(units=50, return_sequences=True))
# adding the third dropout layer
regressor.add(Dropout(rate=0.2))
# adding fourth long short term memory
regressor.add(LSTM(units=50, return_sequences=False))
# adding fourth dropout layer for fourth long short term memory
regressor.add(Dropout(rate=0.2))
# adding output layer for the RNN
regressor.add(Dense(units=1))
# compiling the regressor
regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# fitting the RNN with the dataset
regressor.fit(X_train, y_train, batch_size=32, epochs=100)
regressor.save(MODEL_DIR, overwrite=True)
# importing the testset
dataset_test = pd.read_csv(TEST_DIR)
# getting the real stock price
real_stock_price = dataset_test.iloc[:, 1:2].values
# structuring the prediction data
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# reshaping the inputs to correct format
inputs = inputs.reshape(-1, 1)
# scaling the inputs
inputs = sc.transform(inputs)
# reshaping the inputs
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
# converting the array to 3d
X_test = np.reshape(X_test, X_test.shape[0], X_test.shape[1])
# predicting the results using the regressor
predicted_price = regressor.predict(X_test)
# rescaling the array or results
predicted_stock_price = sc.inverse_transform(predicted_price)
