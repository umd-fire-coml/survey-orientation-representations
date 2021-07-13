import math
import torch
import numpy as np

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

# generates and returns a position encoding matrix
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
    return pe



if __name__=="__main__":
    d_model = 4
    height = 3
    width = 5

    print(f'positionalencoding2d shape: {positionalencoding2d(d_model, height, width).shape}')
    print(f'positionalencoding2d: {positionalencoding2d(d_model, height, width)}')
    print('----')
    print(f'get_2d_pos_enc shape:{get_2d_pos_enc(height, width, d_model).shape}')
    print(f'get_2d_pos_enc:{get_2d_pos_enc(height, width, d_model)}')
