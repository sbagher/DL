# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 03, Chapter: 07, Book: "Python Machine Learning By Example"

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDRegressor
import tensorflow as tf

def compute_prediction(X, weights):
    """
    Compute the prediction y_hat based on current weights
    """
    predictions = np.dot(X, weights)
    return predictions

def update_weights_gd(X_train, y_train, weights, learning_rate):
    """
    Update weights by one step and return updated wights
    """
    predictions = compute_prediction(X_train, weights)
    weights_delta = np.dot(X_train.T, y_train - predictions)
    m = y_train.shape[0]
    weights += learning_rate / float(m) * weights_delta
    return weights

def compute_cost(X, y, weights):
    """
    Compute the cost J(w)
    """
    predictions = compute_prediction(X, weights)
    cost = np.mean((predictions - y) ** 2 / 2.0)
    return cost
def train_linear_regression(X_train, y_train, max_iter, learning_rate, fit_intercept=False):
    """
    Train a linear regression model with gradient descent, and return trained model
    """
    if fit_intercept:
        intercept = np.ones((X_train.shape[0], 1))
        X_train = np.hstack((intercept, X_train))
    weights = np.zeros(X_train.shape[1])
    for iteration in range(max_iter):
        weights = update_weights_gd(X_train, y_train, weights, learning_rate)
        # Check the cost for every 100 (for example) iterations
        if iteration % 500 == 0:
            print(compute_cost(X_train, y_train, weights))
    return weights

def predict(X, weights):
    if X.shape[1] == weights.shape[0] - 1:
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack((intercept, X))
    return compute_prediction(X, weights)

diabetes = datasets.load_diabetes()
print("\nResult of 'Implementing linear regression from scratch' section")
print(diabetes.data.shape)
num_test = 30
X_train = diabetes.data[:-num_test, :]
y_train = diabetes.target[:-num_test]
weights = train_linear_regression(X_train, y_train, max_iter=5000, learning_rate=1, fit_intercept=True)
X_test = diabetes.data[-num_test:, :]
y_test = diabetes.target[-num_test:]
predictions = predict(X_test, weights)
print(predictions)
print(y_test)

print("\nResult of 'Implementing linear regression with scikit-learn' section")
regressor = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=0.01, max_iter=1000)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
print(predictions)
print(y_test)

print("\nResult of 'Implementing linear regression with TensorFlow' section")
layer0 = tf.keras.layers.Dense(units=1, input_shape=[X_train.shape[1]])
model = tf.keras.Sequential(layer0)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(1))
model.fit(X_train, y_train, epochs=100, verbose=True)
predictions = model.predict(X_test)[:, 0]
print(predictions)
print(y_test)
