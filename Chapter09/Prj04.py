# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 02, Chapter: 09, Book: "Python Machine Learning By Example"

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import names

categories_3 = ['talk.religion.misc', 'comp.graphics', 'sci.space']
groups_3 = fetch_20newsgroups(categories=categories_3)

all_names = set(names.words())
count_vector_sw = CountVectorizer(stop_words="english", max_features=500)
lemmatizer = WordNetLemmatizer()
data_cleaned = []
for doc in groups_3.data:
    doc = doc.lower()
    doc_cleaned = ' '.join(lemmatizer.lemmatize(word)
                           for word in doc.split()
                           if word.isalpha() and word not in all_names)
    data_cleaned.append(doc_cleaned)
data_cleaned_count_3 = count_vector_sw.fit_transform(data_cleaned)
print(count_vector_sw.get_feature_names_out())

tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
data_tsne = tsne_model.fit_transform(data_cleaned_count_3.toarray())

plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups_3.target)
plt.show()