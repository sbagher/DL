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

print(count_vector.get_feature_names_out())

data_cleaned = []
for doc in groups.data:
    doc_cleaned = ' '.join(word for word in doc.split() if word.isalpha())
    data_cleaned.append(doc_cleaned)

data_count = count_vector.fit_transform(data_cleaned)
print(count_vector.get_feature_names_out())

from sklearn.feature_extraction import stop_words
print(stop_words.ENGLISH_STOP_WORDS)