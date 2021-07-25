import tensorflow as tf
from orientation_converters import multibin_to_radians
from add_output_layers import LAYER_OUTPUT_NAME_SINGLE_BIN, LAYER_OUTPUT_NAME_TRICOSINE, LAYER_OUTPUT_NAME_ALPHA_ROT_Y, LAYER_OUTPUT_NAME_MULTIBIN, LAYER_OUTPUT_NAME_VOTING_BIN
from tensorflow.keras.losses import mean_squared_error as l2_loss


def __loss_tricosine(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_tricosine = {LAYER_OUTPUT_NAME_TRICOSINE: __loss_tricosine}
loss_tricosine_weights = {LAYER_OUTPUT_NAME_TRICOSINE: 1.0}


def __loss_alpha_rot_y(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_alpha_rot_y = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: __loss_alpha_rot_y}
loss_alpha_rot_y_weights = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: 1.0}


def __loss_multibin(y_true, y_pred):

    loss_conf = tf.reduce_sum(l2_loss(y_true[..., 2:], y_pred[..., 2:]), 1)

    loss_orientation = l2_loss(y_true[..., 0], y_pred[..., 0]) + l2_loss(y_true[..., 1], y_pred[..., 1])

    # print(f'shape of loss_conf:{loss_conf.shape}\nshape of loss_orientation: {loss_orientation.shape}')

    return loss_conf + loss_orientation

loss_multibin = {LAYER_OUTPUT_NAME_MULTIBIN: __loss_multibin}
loss_multibin_weights = {LAYER_OUTPUT_NAME_MULTIBIN: 1.0}


def __loss_single_bin(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_single_bin = {LAYER_OUTPUT_NAME_SINGLE_BIN: __loss_single_bin}
loss_single_bin_weights = {LAYER_OUTPUT_NAME_SINGLE_BIN: 1.0}


def __loss_voting_bin(y_true, y_pred):
    return l2_loss(y_true, y_pred)

loss_voting_bin = {LAYER_OUTPUT_NAME_VOTING_BIN: __loss_voting_bin}
loss_voting_bin_weights = {LAYER_OUTPUT_NAME_VOTING_BIN: 1.0}



def get_loss_params(orientation):
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
    else:
        raise Exception('Incorrect orientation type for loss function')
