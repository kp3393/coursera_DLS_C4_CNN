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