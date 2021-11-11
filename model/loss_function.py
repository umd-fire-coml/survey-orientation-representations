from pandas.tseries.offsets import YearBegin
import tensorflow as tf
from orientation_converters import multibin_to_radians, angle_normed_to_radians
from add_output_layers import LAYER_OUTPUT_NAME_SINGLE_BIN, LAYER_OUTPUT_NAME_TRICOSINE, LAYER_OUTPUT_NAME_ALPHA_ROT_Y, LAYER_OUTPUT_NAME_MULTIBIN, LAYER_OUTPUT_NAME_VOTING_BIN
from tensorflow.keras.losses import mean_squared_error as l2_loss


def __loss_tricosine(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_tricosine = {LAYER_OUTPUT_NAME_TRICOSINE: __loss_tricosine}
loss_tricosine_weights = {LAYER_OUTPUT_NAME_TRICOSINE: 1.0}


def __loss_alpha_rot_y_l2(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_alpha_rot_y = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: __loss_alpha_rot_y_l2}
loss_alpha_rot_y_weights = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: 1.0}


def __loss_alpha_rot_y_angular_normed(y_true, y_pred):
    y_true_rad = angle_normed_to_radians(y_true)
    y_pred_rad = angle_normed_to_radians(y_pred)
    y_true_vector = tf.transpose(tf.stack([tf.cos(y_true_rad), tf.sin(y_true_rad)]))
    y_pred_vector = tf.transpose(tf.stack([tf.cos(y_pred_rad), tf.sin(y_pred_rad)]))
    # perform dot product
    dot_producted = tf.reduce_sum(tf.multiply(y_true_vector, y_pred_vector), 1)
    loss = dot_producted / (tf.norm(y_true_vector, axis=1)*tf.norm(y_pred_vector, axis=1))
    # print(f'y_true_vector: {y_true_vector.numpy()}')
    # print(f'norm of GT:{tf.norm(y_true_vector, axis=1).numpy()}')
    return 1-loss

def __loss_alpha_rot_y_angular(y_true, y_pred):
    y_true_rad = y_true
    y_pred_rad = y_pred
    y_true_vector = tf.transpose(tf.stack([tf.cos(y_true_rad), tf.sin(y_true_rad)]))
    y_pred_vector = tf.transpose(tf.stack([tf.cos(y_pred_rad), tf.sin(y_pred_rad)]))
    # perform dot product
    dot_producted = tf.reduce_sum(tf.multiply(y_true_vector, y_pred_vector), 1)
    loss = dot_producted / (tf.norm(y_true_vector, axis=1)*tf.norm(y_pred_vector, axis=1))
    # print(f'y_true_vector: {y_true_vector.numpy()}')
    # print(f'norm of GT:{tf.norm(y_true_vector, axis=1).numpy()}')
    return 1-loss

loss_alpha_rot_y_angular = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y:__loss_alpha_rot_y_angular_normed}
loss_alpha_rot_y_angular_weights = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: 1.0}

def __loss_multibin(y_true, y_pred):
    #loss_conf = tf.reduce_sum(l2_loss(y_true[..., 2:], y_pred[..., 2:]), 1)
    loss_conf = l2_loss(y_true[..., 2], y_pred[..., 2])
    #loss_conf = tf.reduce_max(loss_conf, 1)
    # 
    loss_orientation = l2_loss(y_true[..., 0], y_pred[..., 0]) + l2_loss(y_true[..., 1] , y_pred[..., 1])
    # return tf.reduce_sum(loss_conf,1) + loss_orientation
    return loss_conf + loss_orientation


loss_multibin = {LAYER_OUTPUT_NAME_MULTIBIN: __loss_multibin}
loss_multibin_weights = {LAYER_OUTPUT_NAME_MULTIBIN: 1.0}


def __loss_single_bin_l2(y_true, y_pred):
    return l2_loss(y_true, y_pred)

def __loss_single_bin_angular(y_true, y_pred):
    pass

loss_single_bin = {LAYER_OUTPUT_NAME_SINGLE_BIN: __loss_single_bin_l2}
loss_single_bin_weights = {LAYER_OUTPUT_NAME_SINGLE_BIN: 1.0}


def __loss_voting_bin(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_voting_bin = {LAYER_OUTPUT_NAME_VOTING_BIN: __loss_voting_bin}
loss_voting_bin_weights = {LAYER_OUTPUT_NAME_VOTING_BIN: 1.0}



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
    elif use_angular_loss:
        return loss_alpha_rot_y_angular, loss_alpha_rot_y_angular_weights
    else:
        raise Exception('Incorrect orientation type for loss function')
    