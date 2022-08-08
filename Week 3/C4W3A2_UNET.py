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


"==========########## Load and Split the Data ##########=========="
ROOT_DIR = os.getcwd()
image_path = os.path.join(ROOT_DIR, 'Data', 'C4W3A2', 'CARLA', 'CameraRGB')
mask_path = os.path.join(ROOT_DIR, 'Data', 'C4W3A2', 'CARLA', 'CameraSeg')
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)

image_list = [os.path.join(image_path, i) for i in image_list]
mask_list = [os.path.join(mask_path, i) for i in mask_list]

# Investigating images
img_nr = 5
img = imageio.imread(image_list[img_nr])
print('Image dimensions are ', img.shape)

mask = imageio.imread(mask_list[img_nr])
print('Mask dimensions are ', mask.shape)
print(np.unique(mask[:, :, 0]))
print(np.unique(mask[:, :, 1]))
print(np.unique(mask[:, :, 2]))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Image')
ax[1].imshow(mask[:, :, 0])
ax[1].set_title('Segmentation')
plt.show()