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

tf.random.set_seed(42)
model = models.Sequential()
embedding_size = 32
model.add(layers.Embedding(vocab_size, embedding_size))
model.add(layers.LSTM(50, return_sequences=True, dropout=0.2))
model.add(layers.LSTM(50, dropout=0.2))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

optimizer = optimizers.Adam(learning_rate=0.003)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
batch_size = 64
n_epoch = 7
model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch, validation_data=(X_test, y_test))

acc = model.evaluate(X_test, y_test, verbose=0)[1]
print('Test accuracy with stacked LSTM:', acc)

#---------------------------------------------------
def train_test_model(hparams):
    model = models.Sequential()
    model.add(layers.Conv2D(hparams[HP_NUM_UNITS1], (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(hparams[HP_NUM_UNITS2], (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(hparams[HP_NUM_UNITS3], activation='relu'))
    model.add(layers.Dense(hparams[HP_NUM_UNITS4], activation='softmax'))

    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_generator, validation_data=(X_test, test_labels), epochs=hparams[HP_EPOCHS], batch_size=40)

    _, accuracy = model.evaluate(X_test, test_labels, verbose=2)

    return accuracy

def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams(hparams)
        accuracy = train_test_model(hparams)
        tf.summary.scalar('accuracy', accuracy, step=1)

HP_EMBEDDING_SIZE = hp.HParam('embedding_size', hp.Discrete([16,32,48])) 
HP_LSTM1 = hp.HParam('1st LSTM layer', hp.Discrete([32,64,128]))
HP_LSTM2 = hp.HParam('2nd LSTM layer', hp.Discrete([32,64,128]))
HP_DROPOUT1 = hp.HParam('1st layer dropout', hp.Discrete([0.2,0.5]))
HP_DROPOUT2 = hp.HParam('2nd layer dropout', hp.Discrete([0.2,0.5]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.003, 0.3))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10, 20]))

session_num = 0
for first in HP_NUM_UNITS1.domain.values:
    for second in HP_NUM_UNITS2.domain.values:
        for third in HP_NUM_UNITS3.domain.values:
            for fourth in HP_NUM_UNITS4.domain.values:
                for epoch in HP_EPOCHS.domain.values:
                    hparams = {
                        HP_NUM_UNITS1: first, 
                        HP_NUM_UNITS2: second, 
                        HP_NUM_UNITS3: third, 
                        HP_NUM_UNITS4: fourth, 
                        HP_EPOCHS: epoch, 
                        }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run(hparams, 'logs/hparam_tuning/' + run_name)
                    session_num += 1
