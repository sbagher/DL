# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 10, Book: "Python Machine Learning By Example"

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data
y = iris.target
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

