# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 03, Chapter: 08, Book: "Python Machine Learning By Example"
import numpy as np
from sklearn.neural_network import MLPRegressor
import pandas as pd

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

nn_scikit = MLPRegressor(hidden_layer_sizes=(16, 8),
                         activation='relu', solver='adam', 
                         learning_rate_init=0.001, 
                         random_state=42, max_iter=2000)
