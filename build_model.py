from tensorflow.keras import Input, Model
from backbone_xception import Xception_model
from add_output_layers import add_output_layers
from data_processing import CROP_RESIZE_H, CROP_RESIZE_W

def build_model(orientation, img_h = CROP_RESIZE_H, img_w = CROP_RESIZE_W, n_channel = 3):
    inputs = Input(shape=(img_h, img_w, n_channel))
    x = Xception_model(inputs, pooling='avg')
    x = add_output_layers(orientation, x)
    model = Model(inputs=inputs, outputs=x)
    return model