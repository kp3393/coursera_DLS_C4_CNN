"""
Implementation of the Neural Style Transfer.

1. Implement the neural style transfer algorithm
2. Generate novel artistic images using your algorithm
3. Define the style cost function for Neural Style Transfer
4. Define the content cost function for Neural Style Transfer

"""

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import typing
import tensorflow as tf
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from tensorflow.python.framework.ops import EagerTensor
import pprint

"#####====== Transfer Learning. Loading pre trained model =====#####"


img_size = 400
vgg = keras.applications.VGG19(include_top=False,
                               input_shape=(img_size, img_size, 3))
vgg.trainable = False

"#####====== Neural Style Transfer: Cost functions =====#####"


ROOT_DIR = os.getcwd()
img_dir = os.path.join(ROOT_DIR, 'images')
fname = os.path.join(img_dir, 'louvre_small.jpg')
content_image = Image.open(fname)

# see the content image
# plt.imshow(content_image)
# plt.show()

fname = os.path.join(img_dir, 'monet.jpg')
style_image = Image.open(fname)

# see the style image
# plt.imshow(style_image)
# plt.show()


def compute_content_cost(content_output, generated_output):
    """
    Computes the cost function for content image
    :param content_output: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations for content image C
    :param generated_output: tensor of dimension (1, n_H, n_W, n_C), hidden layer activation for generated image G
    :return:
    J_content: scalar, content cost function
    """
    a_C = content_output
    a_G = generated_output

    # getting the dimensions of the images
    m, n_H, n_W, n_C = a_G.shape

    # unrolling the images
    a_C_unrolled = tf.reshape(a_C, shape=[m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])

    # calculating the cost function
    J_content = tf.reduce_sum(tf.square(a_G_unrolled-a_C_unrolled))/(4.0 * n_W *n_H *n_C)

    return J_content


def gram_matrix(A):
    """
    Gram matrix (A * AT) computation for the style cost function
    :param A: matrix of shape (n_C, n_H*n_W)
    :return:
    GA: Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Computes the style cost function for a single layer
    :param a_S: tensor of dimension (1, n_H, n_W, n_C), hidden layer activation for image S
    :param a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer acivation for image G
    :return:
    J_style_layer: tensor representing a scalar value
    """
    # retrieving the dimensions
    m, n_H, n_W, n_C = a_S.shape

    # unrolling them into a 2D matrix
    a_S_unrolled = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))

    # computing the gram matrices
    GS = gram_matrix(a_S_unrolled)
    GG = gram_matrix(a_G_unrolled)

    # computing the style function
    J_style_layer = tf.reduce_sum(tf.square(GS-GG))/(4.0 * ((n_H * n_W * n_C)**2))

    return J_style_layer


# selecting style layers
STYLE_LAYERS = [
    ('block1_conv1', 1.0),
    ('block2_conv1', 0.8),
    ('block3_conv1', 0.7),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.1)]


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several choosen layers
    :param style_image_output: tensorflow model output for style image
    :param generated_image_output: tensorflow model output for generated image
    :param STYLE_LAYERS: python list containing:
                        - the names of the layers to extract style from
                        - weight associated with each one of them
    :return:
    J_style: tensor representing a scalar value
    """
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The first element of the array contains the input layer image, which must not to be used.
    a_S = style_image_output[1:]

    # Set a_G to be the output of the choosen hidden layers.
    # The First element of the list contains the input layer image which must not to be used.
    a_G = generated_image_output[1:]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style

    return J


"#####====== Solving the optimization problem =====#####"


# conditioning the content image
content_image = np.array(content_image.resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

# conditioning the style image
style_image = np.array(style_image.resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

# randomly initialize the generated image
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), 0, 0.8)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)


def get_layer_outputs(vgg, layer_names):
    """
    Creates a vgg model that returns a list of intermediate output values
    :param vgg: tf.keras.Model instance
    :param layer_names: list of layer names
    :return:
    tf.keras.Model
    """
    # layer_names has 'layer' elements in it.
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

print(len(vgg_model_outputs.output))

# saving the content and style output
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style enconder

