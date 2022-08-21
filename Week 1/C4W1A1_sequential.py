"""
Code practise for C4W1A2.

Aim:
---
Create a mood classifer using the TF Keras Sequential API
Build a ConvNet to identify sign language digits using the TF Keras Functional API

Goal:-
----
# - Build and train a ConvNet in TensorFlow for a __binary__ classification problem
# - Explain different use cases for the Sequential and Functional APIs

Model:-
-----
ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

--> [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D):
padding 3, input shape 64 x 64 x 3
--> [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D):
Use 32 7x7 filters, stride 1
--> [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization):
for axis 3
--> [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
--> [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D):
Using default parameters
--> [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
--> Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer:
Apply a fully connected layer with 1 neuron and a sigmoid activation.

"""
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow import keras

import typing

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from cnn_utils import *

"--------######## Loading the dataset and normalizing the dataset ########--------"

# the root working directory independent of the OS
ROOT_DIR = os.getcwd()

# path to traina and test files
train_path = os.path.join(ROOT_DIR, 'datasets', '01_C4W1A2_dataset', 'train_happy.h5')
test_path = os.path.join(ROOT_DIR, 'datasets', '01_C4W1A2_dataset', 'test_happy.h5')

# loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset(train_path, test_path)

# normalizing the dataset
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# reshaping y_test and y_train into column vector from row vector
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# looking at the images
# index = 124
# plt.imshow(X_train[index])
# plt.show()

"--------######## Sequential API  ########--------"


def happyModel(input_shape, pad, kernel_params):
    """
    Creates a sequential model
    :param input_shape: tuple. consisting of the input shape of the data
    :param pad: int. amount of padding around each image on vertical and horizontal dimensions
    :param kernel_params: dict. filter size in 'f', nr of filters in 'n_filter' and stride in 'stride'

    :return: a tensorflow model
    """
    model = tf.keras.Sequential([
        # zero padding
        keras.layers.ZeroPadding2D(padding=pad, input_shape=input_shape, data_format="channels_last"),
        # conv2D
        keras.layers.Conv2D(filters=kernel_params['n_filter'], kernel_size=(kernel_params['f'], kernel_params['f']),
                            name='conv0'),
        # batch norm
        keras.layers.BatchNormalization(axis=3, name='bn0'),
        # relu activation
        keras.layers.ReLU(),
        # max pooling
        keras.layers.MaxPooling2D((2, 2), name='max_pool0'),
        # flatten
        keras.layers.Flatten(),
        # dense
        keras.layers.Dense(1, activation='sigmoid', name='fc')

    ])

    return model


kernel_params = {'f': 7, 'n_filter': 32, 'stride': 1}

# create the model
sequential_model = happyModel(input_shape=X_train.shape[1:], pad=3, kernel_params=kernel_params)

# printing the summary of the model
sequential_model.summary()

# compile the model
sequential_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])

# train and fit the model
sequential_model.fit(X_train, Y_train, epochs=10, batch_size=16)

# evaluate the model
sequential_model.evaluate(X_test, Y_test)