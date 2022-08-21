"""
Code practise for C4W1A2.

Aim:
---
Create a mood classifer using the TF Keras Functional API
Build a ConvNet to identify sign language digits using the TF Keras Functional API

Goal:-
----
# - Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
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
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

import typing

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from cnn_utils import *


"--------######## Loading the dataset and normalizing the dataset ########--------"

# the root working directory independent of the OS
ROOT_DIR = os.getcwd()

# path to traina and test files
train_path = os.path.join(ROOT_DIR, 'datasets', '00_C4W1A1_dataset', 'train_signs.h5')
test_path = os.path.join(ROOT_DIR, 'datasets', '00_C4W1A1_dataset', 'test_signs.h5')

# loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset(train_path, test_path)

# normalizing the dataset
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# reshaping y_test and y_train into column vector from row vector
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
print("Number of classes: "+str(classes))
# looking at the images
# index = 590
# plt.title(Y_train[index])
# plt.imshow(X_train[index])
# plt.show()

"--------######## Functional API  ########--------"


def functional_api(input_shape):
    """
    Creates a sequential model. CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    :param input_shape: tuple. consisting of the input shape of the data

    :return: a tensorflow keras model
    """
    # input shape of the image
    input_img = keras.Input(shape=input_shape)
    # LAYER 1
    # first convolutional layer
    Z1 = keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(1, 1), padding='same')(input_img)
    # first ReLU
    A1 = keras.layers.ReLU()(Z1)
    # first pooling layer
    P1 = keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
    # LAYER 2
    # second convolution layer
    Z2 = keras.layers.Conv2D(filters=16, kernel_size=(2, 2), strides=(1, 1), padding='same')(P1)
    # second ReLU
    A2 = keras.layers.ReLU()(Z2)
    # second maxpool
    P2 = keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
    # flatten layer
    F = keras.layers.Flatten()(P2)
    outputs = keras.layers.Dense(units=6, activation='softmax')(F)

    # bringing it all together
    model = keras.Model(inputs=input_img, outputs=outputs)

    return model


# creating the model
functional_model = functional_api(input_shape=(64, 64, 3))
# compile the model
functional_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
# summarize the model
functional_model.summary()

# fitting the dataset
history = functional_model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

print("*"*10)
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
