"""
Implementation of UNET model.

Aim:-
1. Build your own U-Net
2. Explain the difference between a regular CNN and a U-net
3. Implement semantic image segmentation on the CARLA self-driving car dataset
4. Apply sparse categorical crossentropy for pixelwise prediction

Note:- The part of CARLA data set used in this scipt might differ from the one used in the class.
Dataset link:-
https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180513A.zip
"""
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import typing
import tensorflow as tf
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from test_utils import summary, comparator


"==========########## Loading, associating and preprocessing the Data ##########=========="
# load the dataset
ROOT_DIR = os.getcwd()
image_path = os.path.join(ROOT_DIR, 'Data', 'C4W3A2', 'CARLA', 'CameraRGB')
mask_path = os.path.join(ROOT_DIR, 'Data', 'C4W3A2', 'CARLA', 'CameraSeg')
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)

image_list = [os.path.join(image_path, i) for i in image_list]
mask_list = [os.path.join(mask_path, i) for i in mask_list]

# Investigating images
img_nr = 0
img = imageio.imread(image_list[img_nr])
print('Image dimensions are ', img.shape)

mask = imageio.imread(mask_list[img_nr])
print('Mask dimensions are ', mask.shape)
print('%i classes in the masked image' % np.unique(mask[:, :, 0]).shape[0])

# look at the image and corresponding mask
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img)
# ax[0].set_title('Image')
# ax[1].imshow(mask[:, :, 0])
# ax[1].set_title('Segmentation')
# plt.show()

# associate each image with its corresponding mask and create a tf.data.dataset object
dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))


def process_path(image_path, mask_path):
    """
    Creating a tensor out of the image and the mask
    :param image_path: string, path of the image
    :param mask_path: string, path of the mask
    :return:
    img: tensor of dtype tf.float32
    mask: tensor of unint8
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess(image, mask):
    """
    Resizes the image and the mask to (96, 128). Also, normalizes the image.
    :param image: 3D tensor, output of process_path
    :param mask: 3D tensor, output of the process_path
    :return:
    input_image: resize and normalized input image
    input_mask: resize input mask
    """
    input_image = tf.image.resize(image, (96, 128), method='nearest')
    input_mask = tf.image.resize(mask, (96, 128), method='nearest')

    input_image = input_image / 255.

    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)


def display(display_list):
    """
    Displays an input image, the ground truth and the predicted mask
    :param display_list: list, list of images to be displayed. The expected order is: input image, ground truth
                        and predicted image
    :return:
    Displays the images in a 1x3 format
    """
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# plotting the images from the dataset without pre-processing
for image, mask in image_ds.take(1):
    print(image.shape)
    display([image, mask])

for image, mask in processed_image_ds.take(1):
    print(image.shape)
    display([image, mask])



"==========########## UNET MODEL ##########=========="


def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Convolution block for Unet encoder.
    These are the steps:
    1. Adds 2 Conv2D layers with n_filters filters with kernel_size set to 3,
       kernel_initializer set to 'he_normal', padding set to 'same' and 'relu' activation.
    2. if dropout_prob > 0, then adds a Dropout layer with parameter dropout_prob
    3. if max_pooling is set to True, then adds a MaxPooling2D layer with 2x2 pool size
    :param inputs: tensor, input tensor
    :param n_filters: int, number of filters for the convolution layers
    :param dropout_prob: float, dropout probability
    :param max_pooling: bool, maxpooling to reduce the spatial dimension of the output volume
    :return:
    next_layer, skip_connection: Next layer and skip connection outputs
    """
    # step 1: Two conv2D layers
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    # step 2: Check for the dropout_prob
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    # Check if maxpooling needs to be done before passing to next_layer
    if max_pooling:
        next_layer = MaxPooling2D(2, strides=2)(conv)
    else:
        next_layer = conv
    # get the skip connection layer
    skip_layer = conv

    return next_layer, skip_layer


def upsampling_block(expansive_input, contractive_input, n_filters=13):
    """
    Convolutional upsampling block
    These are the steps:
    1. Perform up sampling (Conv2DTranspose) on the previous layer input (expansive_input)
    2. Merge the upsampled expansive_input with skip_connection input (contractive_input)
    3. Perform 2 Conv2D on the merged output
    :param expansive_input: tensor, input tensor from previous layer
    :param contractive_input: tensor, skip connection
    :param n_filters: int, number of filters from the convolutional layers
    :return:
    conv: Tensor output
    """
    # step 1: perform transpose convolution on the expansive input (i.e. input from the previous layer)
    up = Conv2DTranspose(n_filters,
                         3,
                         strides=2,
                         padding='same')(expansive_input)
    # step 2: merge it with the skip connection input
    merge = concatenate([up, contractive_input], axis=3)
    # step 3: perform convolutions (2 Conv2D) on the merged dataset
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    return conv


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=13):
    """
    The final unet model
    :param input_size: tuple, input shape
    :param n_filters: int, number of filters for the convolution layers
    :param n_classes: number of output classes
    :return:
    model: tf.keras.Model
    """
    # step1: get the input
    inputs = Input(input_size)

    # Step2: Begining of the contracting path/ encoder block
    cblock1 = conv_block(inputs=inputs, n_filters=n_filters)
    # chaining the next_layer output of first conv cblock1 with the input of the next block.
    # Also, double the number of filters from here on
    cblock2 = conv_block(inputs=cblock1[0], n_filters=n_filters*2)
    cblock3 = conv_block(inputs=cblock2[0], n_filters=n_filters*4)
    # for cblock4, apply dropout of 0.3
    cblock4 = conv_block(inputs=cblock3[0], n_filters=n_filters*8, dropout_prob=0.3)
    # for cblock5, have dropout of 0.3 with no max_pooling
    cblock5 = conv_block(inputs=cblock4[0], n_filters=n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Step3: Begining of the expanding path/ decoder path
    # ublock6 --> expansive_block=cblock5[0], contractive_block=cblock4[1], reduce the n_filter by half
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    # ublock7 --> expansive_block=ublock6, contractive_block=cblock3[1], reduce the n_filter by half
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters*4)
    # ublock8 --> expansive_block=ublock7, contractive_block=cblock2[1], reduce the n_filter by half
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters*2)
    # ublock9 --> upsampling_block=ublock8, contractive_block=cblock1[1], reduce the n_filter by half
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    model = keras.Model(inputs=inputs, outputs=conv10)

    return model


"==========########## UNET model training ##########=========="

# Set model parameters and looking at the summary
img_height = 96
img_width = 128
num_channels = 3
filters = 32
n_classes = 13

unet = unet_model((img_height, img_width, num_channels), n_filters=filters, n_classes=n_classes)
unet.summary()

# compiling the model
unet.compile(optimizer='adam',
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['Accuracy'])

# training the model
EPOCHS = 40
VAL_SUBSPLITS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32
processed_image_ds.batch(BATCH_SIZE)
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(processed_image_ds.element_spec)
model_history = unet.fit(train_dataset, epochs=EPOCHS)


def create_mask(pred_mask):
    """
    Uses tf.argmax in the axis of the number of classes to return the index with the largest value and
    merge the prediction into a single image
    :param pred_mask: tensor, creates mask for the
    :return:
    pred_mask: predicted mask from the model
    """
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


# plotting the model accuracy
plt.plot(model_history.history['accuracy'])