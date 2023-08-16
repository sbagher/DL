# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Final Exam, Chapter: 13, Q2, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp

vocab_size = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

word_index = imdb.get_word_index()

maxlen = 200
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

def train_test_model(hparams):
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, hparams[HP_EMBEDDING_SIZE]))
    model.add(layers.LSTM(hparams[HP_LSTM1], hparams[HP_DROPOUT1], return_sequences=True))
    model.add(layers.LSTM(hparams[HP_LSTM2], hparams[HP_DROPOUT2]))
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam(learning_rate=hparams[HP_LEARNING_RATE])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=hparams[HP_EPOCHS], validation_data=(X_test, y_test))

    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]

    return accuracy

def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar('accuracy', accuracy, step=1)

HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([16,32,48])) 
HP_LSTM1 = hp.HParam('1st LSTM layer', hp.Discrete([32,64,96]))
HP_LSTM2 = hp.HParam('2nd LSTM layer', hp.Discrete([32,64,96]))
HP_DROPOUT1 = hp.HParam('1st layer dropout', hp.Discrete([0.2,0.4]))
HP_DROPOUT2 = hp.HParam('2nd layer dropout', hp.Discrete([0.2,0.4]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.002, 0.2))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([7, 14, 21]))

tf.random.set_seed(42)
session_num = 0
for embedding_size in HP_EMBEDDING_SIZE.domain.values:
    for lstm1 in HP_LSTM1.domain.values:
        for lstm2 in HP_LSTM2.domain.values:
            for dropout1 in HP_DROPOUT1.domain.values:
                for dropout2 in HP_DROPOUT2.domain.values:
                    for learning_rate in tf.linspace(HP_LEARNING_RATE.domain.min_value, HP_LEARNING_RATE.domain.max_value, 5):
                        for epoch in HP_EPOCHS.domain.values:
                            hparams = {
                                HP_EMBEDDING_SIZE: embedding_size, 
                                HP_LSTM1: lstm1,
                                HP_LSTM2: lstm2,
                                HP_DROPOUT1: dropout1,
                                HP_DROPOUT2: dropout2,
                                HP_LEARNING_RATE: learning_rate,
                                HP_EPOCHS: epoch,
                                }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run(hparams, 'logs/hparam_tuning/' + run_name)
                            session_num += 1
