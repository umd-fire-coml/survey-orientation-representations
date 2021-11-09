from tensorflow.keras import layers
from tensorflow import math as K
from functools import reduce
from orientation_converters import SHAPE_MULTIBIN, SHAPE_SINGLE_BIN, SHAPE_TRICOSINE, SHAPE_ALPHA_ROT_Y, SHAPE_VOTING_BIN, SHAPE_MULTI_AFFINITY_BIN

LAYER_OUTPUT_NAME_TRICOSINE = 'tricosine_layer_output'
LAYER_OUTPUT_NAME_ALPHA_ROT_Y = 'alpha_rot_y_layer_output'
LAYER_OUTPUT_NAME_MULTIBIN = 'multibin_layer_output'
LAYER_OUTPUT_NAME_VOTING_BIN = 'voting_bin_layer_output'
LAYER_OUTPUT_NAME_SINGLE_BIN = 'single_bin_layer_output'

def add_dense_layers(backbone_layer, output_shape, out_layer_name=''):
    y = layers.Dense(256)(backbone_layer)
    y = layers.LeakyReLU(0.1)(y)
    y = layers.Dropout(0.5)(y)
    # prepare number of outputs
    y = layers.Dense(reduce(lambda x, y: x*y, output_shape))(y) 
    y = layers.Reshape(output_shape, name=out_layer_name)(y)
    return y

def add_output_layers(orientation_type, backbone_layer):
    backbone_layer = layers.Flatten()(backbone_layer)
    if orientation_type == 'multibin':
        return add_dense_layers(backbone_layer, SHAPE_MULTI_AFFINITY_BIN, out_layer_name=LAYER_OUTPUT_NAME_MULTIBIN)
    elif orientation_type == 'tricosine':
        return add_dense_layers(backbone_layer, SHAPE_TRICOSINE, out_layer_name=LAYER_OUTPUT_NAME_TRICOSINE)
    elif orientation_type == 'alpha' or orientation_type == 'rot-y':
        return add_dense_layers(backbone_layer, SHAPE_ALPHA_ROT_Y, out_layer_name=LAYER_OUTPUT_NAME_ALPHA_ROT_Y)
    if orientation_type == 'voting-bin':
        return add_dense_layers(backbone_layer, SHAPE_VOTING_BIN, out_layer_name=LAYER_OUTPUT_NAME_VOTING_BIN)
    if orientation_type == 'single-bin':
        return add_dense_layers(backbone_layer, SHAPE_SINGLE_BIN, out_layer_name=LAYER_OUTPUT_NAME_SINGLE_BIN)
    else:
        raise NameError("Invalid orientation_output_type")
