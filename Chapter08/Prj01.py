# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 08, Book: "Python Machine Learning By Example"
import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def train(X, y, n_hidden, learning_rate, n_iter):
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))
    for i in range(1, n_iter+1):
    Z2 = np.matmul(X, W1) + b1
    A2 = sigmoid(Z2)
    Z3 = np.matmul(A2, W2) + b2
    A3 = Z3
    dZ3 = A3 - y
    dW2 = np.matmul(A2.T, dZ3)
    db2 = np.sum(dZ3, axis=0, keepdims=True)
    dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
    dW1 = np.matmul(X.T, dZ2)
    db1 = np.sum(dZ2, axis=0)
    W2 = W2 - learning_rate * dW2 / m
    b2 = b2 - learning_rate * db2 / m
    W1 = W1 - learning_rate * dW1 / m
    b1 = b1 - learning_rate * db1 / m
    if i % 100 == 0:
    cost = np.mean((y - A3) ** 2)
    print('Iteration %i, training loss: %f' %(i, cost))
    model = {'W1':W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model