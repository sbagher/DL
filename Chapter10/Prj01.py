# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 10, Book: "Python Machine Learning By Example"

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
