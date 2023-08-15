# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 01, Chapter: 13, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

word_index = imdb.get_word_index()

maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

tf.random.set_seed(42)
model = models.Sequential()
embedding_size = 32
model.add(layers.Embedding(vocab_size, embedding_size))
model.add(layers.LSTM(50, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(50, dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

optimizer = optimizers.Adam(llearning_rate=0.003)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
batch_size = 64
n_epoch = 7
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose=0)[1]
print('Test accuracy with stacked LSTM:', acc)