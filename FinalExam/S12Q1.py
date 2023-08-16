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
print(X_train.shape)

def train_test_model(hparams, logdir):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model = Sequential([
            Dense(units=hparams[HP_HIDDEN], activation='relu'),
            Dense(units=1)
            ])
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(hparams[HP_LEARNING_RATE]),
                  metrics=['mean_squared_error']
                  )
    model.fit(X_scaled_train, y_train,
              validation_data=(X_scaled_test, y_test),
              epochs=hparams[HP_EPOCHS], verbose=False,
              callbacks=[tf.keras.callbacks.TensorBoard(logdir),hp.KerasCallback(logdir, hparams),
                         tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=200, verbose=0,mode='auto',)],
                         )
    _, mse = model.evaluate(X_scaled_test, y_test)
    pred = model.predict(X_scaled_test)
    r2 = r2_score(y_test, pred)
    return mse, r2

def run(hparams, logdir):
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_NUM_UNITS3, HP_NUM_UNITS4, HP_EPOCHS],
            metrics=[hp.Metric('mean_squared_error', display_name='mse'), hp.Metric('r2', display_name='r2')],
            )
        mse, r2 = train_test_model(hparams, logdir)
        tf.summary.scalar('mean_squared_error', mse, step=1)
        tf.summary.scalar('r2', r2, step=1)


HP_NUM_UNITS1 = hp.HParam('first convolutional layer', hp.Discrete([32,64,128])) 
HP_NUM_UNITS2 = hp.HParam('second convolutional layer', hp.Discrete([64,128,256]))
HP_NUM_UNITS3 = hp.HParam('third convolutional layer', hp.Discrete([32,64,128]))
HP_NUM_UNITS4 = hp.HParam('hidden layer', hp.Discrete([10,20,30]))
HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10, 20, 30]))

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

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=20, batch_size=40)

test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print('Accuracy on test set:', test_acc)
 
datagen = ImageDataGenerator(height_shift_range=3, horizontal_flip=True)
model_aug = tf.keras.models.clone_model(model)

model_aug.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
train_generator = datagen.flow(X_train, train_labels, seed=42, batch_size=40)
model_aug.fit(train_generator, epochs=50, validation_data=(X_test, test_labels))

test_loss, test_acc = model_aug.evaluate(X_test, test_labels, verbose=2)
print('Accuracy on test set:', test_acc)