# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 02, Chapter: 09, Book: "Python Machine Learning By Example"

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

groups = fetch_20newsgroups()

count_vector = CountVectorizer(max_features=500)
data_count = count_vector.fit_transform(groups.data)
print (data_count.astype)
print(data_count[0].astype)
print(data_count[0])
print(data_count.toarray()[0])
