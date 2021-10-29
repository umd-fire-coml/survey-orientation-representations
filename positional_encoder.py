import math
import numpy as np

# generates and returns a position encoding matrix in numpy
def get_2d_pos_enc(height, width, n_channels):
    """
    :param n_channels: number of pos_enc channels
    :param height: height of the image
    :param width: width of the image
    :return: (height, width, n_channels) position encoding matrix
    """
    pe = np.empty(shape=(n_channels, height, width))

    d_model = int(n_channels / 2)
    div_term = np.exp(np.arange(0., d_model, 2) * -(math.log(10000.0) / d_model)) # (n_channels/2)
    pos_w = np.expand_dims(np.arange(0., width), axis=1) 
    pos_h = np.expand_dims(np.arange(0., height), axis=1)

    pe[0:d_model:2, :, :] = np.expand_dims(np.repeat(np.sin(pos_w * div_term).T, height, axis=0), axis=0)
    pe[1:d_model:2, :, :] = np.expand_dims(np.repeat(np.cos(pos_w * div_term).T, height, axis=0), axis=0)
    pe[d_model::2, :, :] = np.expand_dims(np.repeat(np.sin(pos_h * div_term), width, axis=1), axis=0)
    pe[d_model + 1::2, :, :] = np.expand_dims(np.repeat(np.cos(pos_h * div_term), width, axis=1), axis=0)
    pe = np.moveaxis(pe, 0, -1)
    return pe