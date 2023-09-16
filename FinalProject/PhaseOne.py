# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Final Project, Phase One: Data Cleaning & Test, Poet: Parvin E'tesami, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

training_file = 'poems-utf8.txt'
raw_text = open(training_file, 'r', encoding="utf-8").read()

reshaped_text = arabic_reshaper.reshape(raw_text[:200])
bidi_text = get_display(reshaped_text)
print(bidi_text)

all_words = raw_text.split()
unique_words = list(set(all_words))
print(f'Number of all words: {len(all_words)}')
print(f'Number of unique words: {len(unique_words)}')
n_chars = len(raw_text)
print(f'Total characters: {n_chars}')

chars = sorted(list(set(raw_text)))
n_vocab = len(chars)
print(f'Total vocabulary (unique characters): {n_vocab}')
print(chars)

index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))
print(char_to_index)

seq_length = 160
n_seq = int(n_chars / seq_length)

X = np.zeros((n_seq, seq_length, n_vocab))
Y = np.zeros((n_seq, seq_length, n_vocab))

for i in range(n_seq):
    x_sequence = raw_text[i * seq_length : (i + 1) * seq_length]
    x_sequence_ohe = np.zeros((seq_length, n_vocab))
    for j in range(seq_length):
        char = x_sequence[j]
        index = char_to_index[char]
        x_sequence_ohe[j][index] = 1.
    X[i] = x_sequence_ohe
    y_sequence = raw_text[i * seq_length + 1 : (i + 1) * seq_length + 1]
    y_sequence_ohe = np.zeros((seq_length, n_vocab))
    for j in range(seq_length):
        char = y_sequence[j]
        index = char_to_index[char]
        y_sequence_ohe[j][index] = 1.
    Y[i] = y_sequence_ohe

print (X.shape)
print (Y.shape)
