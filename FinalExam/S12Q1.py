# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Final Exam, Chapter: 12, Q1, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras import layers, models, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))

n_small = 500
X_train = X_train[:n_small]
train_labels = train_labels[:n_small]

datagen = ImageDataGenerator(height_shift_range=3, horizontal_flip=True)
train_generator = datagen.flow(X_train, train_labels, seed=42, batch_size=40)

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

HP_NUM_UNITS1 = hp.HParam('first convolutional layer', hp.Discrete([32,64,128])) 
HP_NUM_UNITS2 = hp.HParam('second convolutional layer', hp.Discrete([64,128,256]))
HP_NUM_UNITS3 = hp.HParam('third convolutional layer', hp.Discrete([32,64,128]))
HP_NUM_UNITS4 = hp.HParam('hidden layer', hp.Discrete([10,20,30]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([20, 50, 70]))

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
