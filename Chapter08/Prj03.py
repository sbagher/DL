# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 08, Book: "Python Machine Learning By Example"
import numpy as np
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow import keras

data_url = "http://lib.stat.cmu.edu/datasets/boston"
# data_url = "boston"
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

tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.Dense(units=20, activation='relu'),
    keras.layers.Dense(units=8, activation='relu'),
    keras.layers.Dense(units=1)
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.02))

model.fit(X_train, y_train, epochs=300)
predictions = model.predict(X_test)[:, 0]

print(predictions)
print(y_test)

print(np.mean((y_test - predictions) ** 2))
