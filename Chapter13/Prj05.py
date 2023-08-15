# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 13, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np

training_file = 'warpeace_input.txt'
raw_text = open(training_file, 'r').read()
raw_text = raw_text[3:3196216]
raw_text = raw_text.lower()

all_words = raw_text.split()
unique_words = list(set(all_words))
n_chars = len(raw_text)

chars = sorted(list(set(raw_text)))
n_vocab = len(chars)

index_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_index = dict((c, i) for i, c in enumerate(chars))

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
