# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 08, Book: "Python Machine Learning By Example"
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
data_url = "boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

num_test = 10 # the last 10 samples as testing set
scaler = preprocessing.StandardScaler()
X_train = data[:-num_test, :]
X_train = scaler.fit_transform(X_train)
y_train = target[:-num_test].reshape(-1, 1)
X_test = data[-num_test:, :]
X_test = scaler.transform(X_test)
y_test = target[-num_test:]

nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8),
                         activation='relu', solver='adam', 
                         learning_rate_init=0.001, 
                         random_state=42, max_iter=2000)

nn_scikit.fit(X_train, y_train.ravel())
predictions = nn_scikit.predict(X_test)
print(predictions)
print(y_test)

print(np.mean((y_test - predictions) ** 2))
