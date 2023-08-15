# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 13, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print('Number of training samples:', len(y_train))
print('Number of positive samples', sum(y_train))
print('Number of test samples:', len(y_test))
print(X_train[0])