# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 10, Book: "Python Machine Learning By Example"

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from copy import deepcopy

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

k = 3
random_index = np.random.choice(range(len(X)), k)
centroids = X[random_index]

def visualize_centroids(X, centroids):
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
    plt.show()

visualize_centroids(X, centroids)

def dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def assign_cluster(x, centroids):
    distances = dist(x, centroids)
    cluster = np.argmin(distances)
    return cluster

def update_centroids(X, centroids, clusters):
    for i in range(k):
        cluster_i = np.where(clusters == i)
        centroids[i] = np.mean(X[cluster_i], axis=0)

tol = 0.0001
max_iter = 100

iter = 0
centroids_diff = 100000
clusters = np.zeros(len(X))

while iter < max_iter and centroids_diff > tol:
    for i in range(len(X)):
    clusters[i] = assign_cluster(X[i], centroids)
    centroids_prev = deepcopy(centroids)