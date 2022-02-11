import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import tensorflow as tf
from orientation_converters import multibin_to_radians, angle_normed_to_radians
from add_output_layers import (
    LAYER_OUTPUT_NAME_SINGLE_BIN,
    LAYER_OUTPUT_NAME_TRICOSINE,
    LAYER_OUTPUT_NAME_ALPHA_ROT_Y,
    LAYER_OUTPUT_NAME_MULTIBIN,
    LAYER_OUTPUT_NAME_VOTING_BIN,
    LAYER_OUTPUT_NAME_EXP_A
)
from tensorflow.keras.losses import mean_squared_error as l2_loss
import numpy as np


def loss_tricosine_(y_true, y_pred):
    return l2_loss(y_true, y_pred)


loss_tricosine = {LAYER_OUTPUT_NAME_TRICOSINE: loss_tricosine_}
loss_tricosine_weights = {LAYER_OUTPUT_NAME_TRICOSINE: 1.0}


def loss_alpha_rot_y_l2_(y_true, y_pred):
    return l2_loss(y_true, y_pred)


loss_alpha_rot_y = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: loss_alpha_rot_y_l2_}
loss_alpha_rot_y_weights = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: 1.0}


def loss_alpha_rot_y_angular_normed_(y_true, y_pred):
    y_true_rad = angle_normed_to_radians(y_true)
    y_pred_rad = angle_normed_to_radians(y_pred)
    y_true_vector = tf.transpose(tf.stack([tf.cos(y_true_rad), tf.sin(y_true_rad)]))
    y_pred_vector = tf.transpose(tf.stack([tf.cos(y_pred_rad), tf.sin(y_pred_rad)]))
    # perform dot product
    dot_producted = tf.reduce_sum(tf.multiply(y_true_vector, y_pred_vector), 1)
    loss = dot_producted / (tf.norm(y_true_vector, axis=1) * tf.norm(y_pred_vector, axis=1))
    return 1 - loss


def loss_alpha_rot_y_angular_(y_true, y_pred):
    y_true_rad = y_true
    y_pred_rad = y_pred
    y_true_vector = tf.transpose(tf.stack([tf.cos(y_true_rad), tf.sin(y_true_rad)]))
    y_pred_vector = tf.transpose(tf.stack([tf.cos(y_pred_rad), tf.sin(y_pred_rad)]))
    # perform dot product
    dot_producted = tf.reduce_sum(tf.multiply(y_true_vector, y_pred_vector), 1)
    loss = dot_producted / (tf.norm(y_true_vector, axis=1) * tf.norm(y_pred_vector, axis=1))
    return 1 - loss


loss_alpha_rot_y_angular = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: loss_alpha_rot_y_angular_normed_}
loss_alpha_rot_y_angular_weights = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: 1.0}

# Current multi affinity loss
def loss_multi_affinity__(y_true, y_pred):
    loss_conf = l2_loss(y_true[..., 2], y_pred[..., 2])
    loss_orientation = l2_loss(y_true[..., 0], y_pred[..., 0]) + l2_loss(
        y_true[..., 1], y_pred[..., 1]
    )
    return loss_conf + loss_orientation
    # return l2_loss(y_true, y_pred)
# def loss_multi_affinity__(y_true, y_pred):

#     loss_conf = tf.reduce_sum(l2_loss(y_true[..., 2:], y_pred[..., 2:]), 1)

#     loss_orientation = l2_loss(y_true[..., 0], y_pred[..., 0]) + l2_loss(y_true[..., 1], y_pred[..., 1])

#     # print(f'shape of loss_conf:{loss_conf.shape}\nshape of loss_orientation: {loss_orientation.shape}')

#     return loss_conf + loss_orientation


loss_multibin = {LAYER_OUTPUT_NAME_MULTIBIN: loss_multi_affinity__}
loss_multibin_weights = {LAYER_OUTPUT_NAME_MULTIBIN: 1.0}


def loss_single_bin_l2_(y_true, y_pred):
    return l2_loss(y_true, y_pred)


def loss_single_bin_angular_(y_true, y_pred):
    pass


loss_single_bin = {LAYER_OUTPUT_NAME_SINGLE_BIN: loss_single_bin_l2_}
loss_single_bin_weights = {LAYER_OUTPUT_NAME_SINGLE_BIN: 1.0}


def loss_voting_bin_(y_true, y_pred):
    return l2_loss(y_true, y_pred)


loss_voting_bin = {LAYER_OUTPUT_NAME_VOTING_BIN: loss_voting_bin_}
loss_voting_bin_weights = {LAYER_OUTPUT_NAME_VOTING_BIN: 1.0}

def loss_exp_A_(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_exp_A = {LAYER_OUTPUT_NAME_EXP_A: loss_exp_A_}
loss_exp_A_weights = {LAYER_OUTPUT_NAME_EXP_A: 1.0}


def get_loss_params(orientation, use_angular_loss):
    if orientation == 'tricosine':
        return loss_tricosine, loss_tricosine_weights
    elif orientation == 'alpha' or orientation == 'rot-y':
        return loss_alpha_rot_y, loss_alpha_rot_y_weights
    elif orientation == 'multibin':
        return loss_multibin, loss_multibin_weights
    elif orientation == 'voting-bin':
        return loss_voting_bin, loss_voting_bin_weights
    elif orientation == 'single-bin':
        return loss_single_bin, loss_single_bin_weights
    elif orientation == 'exp-A':
        return loss_exp_A, loss_exp_A_weights
    elif use_angular_loss:
        return loss_alpha_rot_y_angular, loss_alpha_rot_y_angular_weights
    else:
        raise Exception('Incorrect orientation type for loss function')
