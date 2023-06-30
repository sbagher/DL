# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 03, Chapter: 10, Book: "Python Machine Learning By Example"

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

kmeans_sk = KMeans(n_clusters=3, random_state=42)
kmeans_sk.fit(X)
clusters_sk = kmeans_sk.labels_
centroids_sk = kmeans_sk.cluster_centers_

plt.scatter(X[:, 0], X[:, 1], c=clusters_sk)
plt.scatter(centroids_sk[:, 0], centroids_sk[:, 1], marker='*', s=200, c='#050505')
plt.show()