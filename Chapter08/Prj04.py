# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 08, Book: "Python Machine Learning By Example"
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from tensorflow import keras

data_raw = pd.read_csv('19880101_20191231.csv', index_col='Date')
data = generate_features(data_raw)

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.02))

model.fit(X_train, y_train, epochs=300)
predictions = model.predict(X_test)[:, 0]

print(predictions)
print(y_test)

print(np.mean((y_test - predictions) ** 2))
