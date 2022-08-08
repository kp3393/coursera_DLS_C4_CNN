"""
Implementation of YOLO model.
As the YOLO model is very computationally expensive to train, the pre-trained weights are already loaded for you to use.
Aim:-
1. Detect objects in a car detection dataset
2. Implement non-max suppression to increase accuracy
3. Implement intersection over union
4. Handle bounding boxes, a type of image annotation popular in deep learning

Note:- Couldn't load the yolo.h5 file in tensorflow v2.9.1 and python v3.9 because of Tensorflow bad marshal error
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import typing
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence
    :param boxes: tensor of shape (19, 19, 5, 4). Consists of midpoint of the box and corresponding dimensions.
    :param box_confidence: tensor of shape (19, 19, 5, 1). Confidence score of the class consisting of the object.
    :param box_class_probs: tensor of shape (19, 19, 5, 80). Class score
    :param threshold: float. threshold value for elimination
    :return:
    scores: tensor of shape (None, ), containing the class probability score for selected boxes
    boxes: tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of the selected boxes
    classes: tensor of shape (None, 4), containing the index of the class detected by the selected boxes
    """
    # step 1: computing box_scores = box_confidence * box_class_probs.
    # The resulting tensor will be of shape (19, 19, 5, 80).
    box_scores = box_confidence * box_class_probs

    # box classes (find the class with the maximum score and take a note of them)
    highest_class_index = tf.math.argmax(box_scores, axis=-1)
    highest_class_val = tf.math.reduce_max(box_scores, axis=-1)

    # create a mask with values greater than the threshold
    filtering_mask = (highest_class_val >= threshold)

    # now apply the mask to the box_class_score, boxes
    scores = tf.boolean_mask(highest_class_val, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(highest_class_index, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """
    Implementation of intersection over union between box1 and box2
    :param box1: list, first box with coordinates (box1_x1, box1_y1, box1_x2, box1_y2)
    :param box2: list, second box with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    :return: iou: float with intersection over union calculation
    """
    # unpacking the list of coordinates
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # calculating intersection coordinates
    # left top coordinates
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    # bottom right coordinates
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    # calculating the area
    intersection_height = max(0, yi2-yi1)
    intersection_width = max(0, xi2-xi1)
    intersection_area = intersection_width * intersection_height

    # calculating the union
    box1_area = (box1_x2-box1_x1) * (box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1) * (box2_y2-box2_y1)
    union_area = box1_area + box2_area - intersection_area

    # computing iou
    iou = intersection_area/union_area

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Non-max supression using tensorflow libraries
    :param scores: tensor of shape (None,), output of yolo_filter_boxes()
    :param boxes: tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size
    :param classes: tensor of shape (None,), output of yolo_filter_boxes()
    :param max_boxes: integer, maximum number of predicted boxes you'd like
    :param iou_threshold: real value, "intersection over union" threshold used for NMS filtering
    :return:
    scores: tensor of shape (, None), predicted score for each box
    boxes: tensor of shape (4, None), predicted box coordinates
    classes: tensor of shape (, None), predicted class for each box
    """
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')
    # tf.image.non_max_supression will give us indices of the boxes which satisfy our phenomena
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    # use the indices from nms_indices to slice out the data we need. Read the documentation of
    # tf.image.non_max_supression and tf.gather
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_boxes_to_corners(box_xy, box_wh):
    """
    Convert YOLO box predictions to bounding box corners.
    :param box_xy: tensor of shape (None, 19, 19, 5, 2). x, y coordinate of the center of the box
    :param box_wh: tensor of shape (None, 19, 19, 5, 2). width and height of the box
    :return:
    box corners' coordinates (x1, y1, x2, y2) to fit the input of yolo_filter_boxes
    """
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Bringing it all together. Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores,
    box coordinates and classes.
    :param yolo_outputs: output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                         box_xy: tensor of shape (None, 19, 19, 5, 2)
                         box_wh: tensor of shape (None, 19, 19, 5, 2)
                         box_confidence: tensor of shape (None, 19, 19, 5, 1)
                         box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    :param image_shape: tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.)
                        (has to be float32 dtype)
    :param max_boxes: integer, maximum number of predicted boxes
    :param score_threshold: real value, if [the highest class probability score < threshold],
                            then get rid of the corresponding box
    :param iou_threshold: real value, "intersection over union" threshold used for NMS filtering
    :return:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    # unpacking the output of the encoding model
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    # convert boxes (box_xy, box_wh) to corners (x1, y1, x2, y2)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # get the classes with the maximum scores
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)

    # Scale boxes back to original image shape (720, 1280 or whatever)
    boxes = scale_boxes(boxes, image_shape)  # Network was trained to run on 608x608 images

    # perform NMS
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


