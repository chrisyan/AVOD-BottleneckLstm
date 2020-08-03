#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:30:50 2019

@author: emec
"""
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os


def tf_iou(boxes1, boxes2):
    """
    from https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d

    Calculates vectorized intersection area over union area of bounding boxes among each other

    boxes1 : bounding boxes with a shape of (m,4) in the [x_min,y_min,x_max,y_max] format
    boxes2 : bounding boxes with a shape of (n,4) in the [x_min,y_min,x_max,y_max] format

    Returns Intersection over Union (IoU) values of bounding boxes with a shape of (m,n)

    """
    x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1)

    xA = tf.maximum(x11, tf.transpose(x21))
    yA = tf.maximum(y11, tf.transpose(y21))
    xB = tf.minimum(x12, tf.transpose(x22))
    yB = tf.minimum(y12, tf.transpose(y22))

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)

    boxAArea = (x12 - x11) * (y12 - y11)
    boxBArea = (x22 - x21) * (y22 - y21)

    iou = interArea / (boxAArea + tf.transpose(boxBArea) - interArea)

    return iou, boxAArea, boxBArea, interArea


def test_tf_iou():
    """
    # To show boxes on a random image
    im_rand = np.random.randint(0,255,size=(25,25))
    show_image(im_rand.astype(np.uint8),boxes_proposal,save='deneme',line_width=1)

    """
    ## Hand-made box definitions
    # artificial bounding box proposals
    boxes_proposal = np.array(
        [[2, 2, 5, 7], [1, 5, 7, 8], [15, 2, 21, 6], [13, 4, 19, 7], [3, 9, 6, 14], [7, 9, 13, 13], \
         [17, 10, 21, 16], [4, 17, 10, 21], [12, 18, 15, 23], [19, 18, 23, 23]])
    # artificial ground-truth box proposals
    boxes_gt = np.array(
        [[4, 4, 9, 8], [12, 3, 20, 6], [4, 10, 13, 13], [16, 9, 20, 15], [3, 19, 7, 23], [8, 18, 13, 23]])

    proposal_areas = np.array([15, 18, 24, 18, 15, 24, 24, 24, 15, 20])
    gt_areas = np.array([20, 24, 27, 24, 16, 25])

    # hand-calculated intersection areas between boxes
    intersection_mat = np.array([[3, 0, 0, 0, 0, 0], [9, 0, 0, 0, 0, 0], [0, 15, 0, 0, 0, 0], [0, 12, 0, 0, 0, 0], \
                                 [0, 0, 6, 0, 0, 0], [0, 0, 18, 0, 0, 0], [0, 0, 0, 15, 0, 0], [0, 0, 0, 0, 6, 6], \
                                 [0, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0]])
    # hand-calculated union areas between boxes
    union_mat = np.array(
        [[32, 39, 41, 39, 31, 40], [29, 42, 45, 42, 34, 43], [44, 33, 51, 48, 40, 49], [38, 30, 45, 42, 34, 43], \
         [35, 39, 36, 39, 31, 40], [44, 48, 33, 48, 40, 49], [44, 48, 51, 33, 40, 49], [44, 48, 51, 48, 34, 43], \
         [35, 39, 42, 39, 31, 35], [40, 44, 47, 44, 36, 45]])

    # Intersection over Union value calculated by using hand-calculated values
    int_over_union = intersection_mat / union_mat

    # assume that the threshold for objectness is 0.39 and for background 0.16
    # 0: bg, 1: obj, -1: nothing
    objectness = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0])
    bg_gt = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    nothing_arr = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    object_boxes_proposal = np.array([[1, 5, 7, 8], [15, 2, 21, 6], [13, 4, 19, 7], [7, 9, 13, 13], \
                                      [17, 10, 21, 16], [4, 17, 10, 21], [12, 18, 15, 23]])

    object_boxes_gt = np.array([[4, 4, 9, 8], [12, 3, 20, 6], [12, 3, 20, 6], [4, 10, 13, 13], \
                                [16, 9, 20, 15], [3, 19, 7, 23], [8, 18, 13, 23]])

    bg_boxes_proposal = np.array([[2, 2, 5, 7], [3, 9, 6, 14], [19, 18, 23, 23]])

    # Test the functions
    iou, boxA, boxB, interAB = tf_iou(boxes_proposal, boxes_gt)

    with tf.Session() as sess:
        func_iou, func_boxA, func_boxB, func_interAB = sess.run([iou, boxA, boxB, interAB])

    print("Proposal shape match: ", len(proposal_areas) == len(np.reshape(func_boxA, (-1))))
    print("Ground-truth shape match: ", len(gt_areas) == len(np.reshape(func_boxB, (-1))))

    print("Proposal area match: ", (proposal_areas == np.reshape(func_boxA, (-1))).all())
    print("Ground-truth area match: ", (gt_areas == np.reshape(func_boxB, (-1))).all())

    print("Intersection match: ", (intersection_mat == func_interAB).all())
    print("IoU match: ", (int_over_union == func_iou).all())

test_tf_iou()