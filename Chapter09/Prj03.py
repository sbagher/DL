# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 02, Chapter: 09, Book: "Python Machine Learning By Example"

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import matplotlib.pyplot as plt

groups = fetch_20newsgroups()

from nltk.corpus import names
all_names = set(names.words())
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                           for word in doc.split()
                           if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)
data_cleaned_count = count_vector_sw.fit_transform(data_cleaned)