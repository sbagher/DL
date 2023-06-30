# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 10, Book: "Python Machine Learning By Example"

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target
k_list = list(range(1, 7))
sse_list = [0] * len(k_list)

for k_ind, k in enumerate(k_list):
    kmeans = KMeans(n_clusters=k, random_state=42, init='random', n_init='auto')
    kmeans.fit(X)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sse = 0
    for i in range(k):
        cluster_i = np.where(clusters == i)
        sse += np.linalg.norm(X[cluster_i] - centroids[i])
    print('k={}, SSE={}'.format(k, sse))
    sse_list[k_ind] = sse