# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 06, Chapter: 10, Book: "Python Machine Learning By Example"

from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import NMF

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    ]
groups = fetch_20newsgroups()
labels = groups.target
label_names = groups.target_names
def is_letter_only(word):
    for char in word:
        if not char.isalpha():
            return False
    return True

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word) for word in doc.split() if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)

count_vector = CountVectorizer(stop_words="english", max_features=None, max_df=0.5, min_df=2)

t = 20
nmf = NMF(n_components=t, random_state=42)
data = count_vector.fit_transform(data_cleaned)
nmf.fit(data)
print (nmf.components_)

terms = count_vector.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic {}:" .format(topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[-10:]]))

tfidf_vector = TfidfVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2)
nmf = NMF(n_components=t, random_state=42)
data = tfidf_vector.fit_transform(data_cleaned)
nmf.fit(data)
print (nmf.components_)

terms = tfidf_vector.get_feature_names_out()
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic {}:" .format(topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[-10:]]))
