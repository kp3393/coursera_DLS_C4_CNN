"""
Implementation of  MobileNetV2.

Aim:-  transfer learning on a pre-trained CNN to build an Alpaca/Not Alpaca classifier!

Learn to:
Create a dataset from a directory
Preprocess and augment data using the Sequential API
Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
Fine-tune a classifier's final layers to improve accuracy

"""

import matplotlib.pyplot as plt
import numpy as np
import os
import typing

import tensorflow as tf
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from keras.utils import image_dataset_from_directory
from keras.layers import RandomFlip, RandomRotation


"==========########## Create the Dataset and Split it into Training and Validation Sets ##########=========="


BATCH_SIZE = 32
IMG_SIZE = (160, 160)
ROOT_DIR = os.getcwd()
data_folder = os.path.join(ROOT_DIR, 'Data', 'Data_MobileNet')

train_dataset = image_dataset_from_directory(data_folder,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.2,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(data_folder,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=42)
# retreving the class names
class_names = train_dataset.class_names

# plotting figure
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


"==========########## Preprocess and Augment Training Data ##########=========="


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


def data_augmenter():
    """
    Creates a Sequential model composed of 2 layers
    :return:
    tf.keras.Sequential
    """
    # create an instance of sequential
    data_augmentation = keras.Sequential()
    # add random flip
    data_augmentation.add(RandomFlip('horizontal'))
    data_augmentation.add(RandomRotation(0.2))

    return data_augmentation

# create an instance of the function
data_augmentation = data_augmenter()

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0]/255)
        plt.axis('off')
plt.show()


"==========########## Using MobileNetV2 for Transfer Learning ##########=========="


# preprocess images to match the input requirement for mobilenet_v2
preprocess_input = keras.applications.mobilenet_v2.preprocess_input


def alpaca_model(image_shape=IMG_SIZE, data_augmentation=data_augmenter()):
    """
    Defines a tf.keras model for binary classification out of the MobileNetV2 model
    :param image_shape: tuple, image width and height
    :param data_augmentation: data augmentation function
    :return:
    tf.keras.model
    """
    # making it a three-dimensional (width, height, channel) array from a two-dimensional array (width, height)
    input_shape = image_shape + (3,)
    # in traning the base model, DO NOT INCLUDE THE TOP LAYER
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                             include_top=False,
                                                             weights='imagenet')
    # Freezing the base model by making it non-trainable
    base_model.trainable = False
    # Creating an input layer for the base model
    inputs = keras.Input(shape=input_shape)
    # applying data augmentation to the inputs
    x = data_augmentation(inputs)
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(x)
    # set base layer training to 'false'
    x = base_model(x, training=False)
    # apply global average pooling to summarize the info in each channel
    x = keras.layers.GlobalAvgPool2D()(x)
    # dropout with probability of 0.2
    x = keras.layers.Dropout(0.2)(x)

    # prediction layer with one neuron
    prediction_layer = keras.layers.Dense(1)
    outputs = prediction_layer(x)

    # bringing it all togehter
    model = keras.Model(inputs, outputs)

    return model


# creating model with correct paramaters
model_alpaca = alpaca_model(IMG_SIZE, data_augmentation)

# compiling the model
base_learning_rate = 0.01
model_alpaca.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['accuracy'])

# fitting it on the data
initial_epochs = 5
history = model_alpaca.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

print(model_alpaca.summary())

# plotting accuracy and loss
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


"==========########## Fine tuning the model ##########=========="
# get the MobileNet model out
base_model = model_alpaca.layers[4]
# make it trainable
base_model.trainable = True
print(len(base_model.layers))

# Fine-tune from selected layer
fine_tune_from = 117

# freeze all the layers before this layer
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

# defining the BinaryCrossentropy loss function
loss_function = keras.losses.BinaryCrossentropy(from_logits=True)
# selecting the optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.003*base_learning_rate)
# Evaluation metric
metrics = ['accuracy']

model_alpaca.compile(loss=loss_function,
                     optimizer=optimizer,
                     metrics=metrics)

fine_tune_epochs = 5
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model_alpaca.fit(train_dataset,
                                epochs=total_epochs,
                                initial_epoch=history.epoch[-1],
                                validation_data=validation_dataset)

# plotting accuracy and loss
acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0, 1])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1, initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()





