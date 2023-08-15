# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 04, Chapter: 13, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

training_file = 'warpeace_input.txt'
raw_text = open(training_file, 'r').read()
raw_text = raw_text[3:3196216]
raw_text = raw_text.lower()

print(raw_text[:200])
all_words = raw_text.split()
unique_words = list(set(all_words))
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