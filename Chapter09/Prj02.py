# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 02, Chapter: 09, Book: "Python Machine Learning By Example"

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

groups = fetch_20newsgroups()

print (groups.keys())

print (groups['target_names'])

print (groups.target)
print (np.unique(groups.target))


sns.distplot(groups.target)
plt.show()

print (groups.data[0])

print (groups.target[0])

print (groups.target_names[groups.target[0]])