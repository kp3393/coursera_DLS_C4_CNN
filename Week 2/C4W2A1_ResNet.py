"""
Implementation of  Residual Networks (ResNets).

Aim:- ResNet model implementation

Implement the basic building blocks of ResNets in a deep neural network using Keras
Put together these building blocks to implement and train a state-of-the-art neural network for image classification
Implement a skip connection in your network

"""
import numpy as np
import scipy.misc
import typing

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from resnets_utils import *
from keras.initializers.initializers_v2 import RandomUniform, GlorotUniform, Constant, Identity
from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow


def identity_block(X, f, filters, training=True, initializer=RandomUniform):
    """
    Implementation of identity block which skips over three layers

    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layer of the main path
    :param training: True. Behave in training mode
                     False. Behave in inference mode
    :param initializer: to set up the initial weight of the layer. Equal to random initializer

    :return:
    X :- out put of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # retrieve number of filters in each layer
    F1, F2, F3 = filters

    # saving the input X to X_shortcut
    X_shortcut = X

    # main path: first layer conv2D --> BN --> ReLU
    X = Conv2D(filters=F1, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # main path: second layer conv2D --> BN --> ReLU
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # main path: third layer conv2D --> BN
    X = Conv2D(filters=F3, kernel_size=1, strides=1, padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # adding with X_shortcut
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True, initializer=GlorotUniform):
    """
    Implementation of the convolutional block
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param s: Integer, specifying the stride to be used
    :param training: True: Behave in training mode
                    False: Behave in inference mode
    :param initializer: to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                        also called Xavier uniform initializer.

    :return:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # MAIN PATH

    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    # SHORTCUT PATH
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid',
                        kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)


    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    :param input_shape: shape of the images of the dataset
    :param classes: integer, number of classes

    :return:
    model: a Model() instance in keras
    """
    # Defining the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=7, strides=2, kernel_initializer=GlorotUniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=2)(X)

    # stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, filters=[128, 128, 512])
    X = identity_block(X, 3, filters=[128, 128, 512])
    X = identity_block(X, 3, filters=[128, 128, 512])

    # stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])

    # stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, 3, filters=[512, 512, 2048])
    X = identity_block(X, 3, filters=[512, 512, 2048])

    # average pooling --> flatten --> FC
    X = AveragePooling2D((2, 2))(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=GlorotUniform(seed=0))(X)

    # create model
    model = Model(inputs=X_input, outputs=X)

    return model

# creating a model
model = ResNet50(input_shape=(64, 64, 3), classes=6)
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load the dataset
ROOT_DIR = os.getcwd()
TRAIN_PATH = os.path.join(ROOT_DIR, 'Data', 'Data_ResNets', 'train_signs.h5')
TEST_PATH = os.path.join(ROOT_DIR, 'Data', 'Data_ResNets', 'test_signs.h5')
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(TRAIN_PATH, TEST_PATH)

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# fitting on the model
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# predictions
preds = model.evaluate(X_test, Y_test)
print('Loss = ', preds[0])
print('Test accuracy = ', preds[1])






