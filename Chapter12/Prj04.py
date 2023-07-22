# Name: Saeed Baghershahi
# Student Number: 102501002
# Class: Deep Learning
# Assignment: Project: 02, Chapter: 12, Book: "Python Machine Learning By Example"

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 

def generate_plot_pics(datagen, original_img, save_prefix):
    folder = 'aug_images'
    try:
        ## if the preview folder does not exist, create
        os.mkdir(folder)
    except:
        ## if the preview folder exists, then remove
        ## the contents (pictures) in the folder
        for item in os.listdir(folder):
            os.remove(folder + "/" + item)
    i = 0
    for batch in datagen.flow(original_img.reshape((1, 28, 28, 1)), batch_size=1, save_to_dir=folder, save_prefix=save_prefix, save_format='jpeg'):
        i += 1
        if i > 2:
            break
    plt.subplot(2, 2, 1, xticks=[],yticks=[])
    plt.imshow(original_img)
    plt.title("Original")
    i = 1
    for file in os.listdir(folder):
        if file.startswith(save_prefix):
            plt.subplot(2, 2, i + 1, xticks=[],yticks=[])
            aug_img = load_img(folder + "/" + file)
            plt.imshow(aug_img)
            plt.title(f"Augmented {i}")
            i += 1
    plt.show()

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

datagen = ImageDataGenerator(horizontal_flip=True)
generate_plot_pics(datagen, train_images[0], 'horizontal_flip')

datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
generate_plot_pics(datagen, train_images[0], 'hv_flip')

datagen = ImageDataGenerator(rotation_range=30)
generate_plot_pics(datagen, train_images[0], 'rotation')

datagen = ImageDataGenerator(width_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_shift')

datagen = ImageDataGenerator(width_shift_range=8, height_shift_range=8)
generate_plot_pics(datagen, train_images[0], 'width_height_shift')

