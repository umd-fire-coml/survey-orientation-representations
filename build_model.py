from tensorflow.keras import Input, Model
from backbone_xception import Xception_model
from add_output_layers import add_output_layers
from data_processing import NORM_H, NORM_W

def build_model(orientation, add_pos_enc: bool):
    n_channel = 6 if add_pos_enc else 3
    inputs = Input(shape=(NORM_H, NORM_W, n_channel))
    x = Xception_model(inputs, pooling='avg')
    x = add_output_layers(orientation, x)
    model = Model(inputs=inputs, outputs=x)
    return model