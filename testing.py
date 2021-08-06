'''Run this file to get all predictions in a json file, outputs to default directory ./preds/'''
# XD the length of import idk i just threw all of these together

import tensorflow as tf
from backbone_xception import Xception_model
from add_output_layers import add_output_layers
from tensorflow.keras import Input, Model
from loss_function import *
import data_processing as dp
import os
import argparse
import time
from datetime import datetime
from collections import namedtuple
from metrics import OrientationAccuracy
from tqdm import tqdm
import numpy as np
from pathlib2 import Path
import visualization
import random
import matplotlib.pyplot as plt
from data_processing import KittiGenerator
from tqdm import tqdm
from os.path import join
import math
import json
import orientation_converters as conv
from serialize import NumpyEncoder,json_numpy_obj_hook

# init code
tf.config.list_physical_devices('GPU')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.8)
            # device_count = {'GPU': 1}
        )
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
parser = argparse.ArgumentParser(description='Testing Model')
parser.add_argument(dest='orientation', type=str,
                    help='Orientation Type of the model. Options are tricosine, alpha, rot_y, multibin')

parser.add_argument('weight_path', type=str, help='Relative path to save weights. Default path is weights')

parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                    help='Define the batch size for training. Default value is 8')


parser.add_argument('--kitti_dir', dest='kitti_dir', type=str, default='dataset',
                    help='path to kitti dataset directory. Its subdirectory should have training/ and testing/. Default path is dataset/')

parser.add_argument('--workers', dest='workers', type=int, default=6,
                    help='amount of worker threads to throw at dataprocessing (more should be better)')

parser.add_argument('--output-dir',dest='output_dir',type= str,default='preds',
                   help='Relative path to store the predictions')

args = parser.parse_args()        
#helper

def loss_func(orientation):
    if orientation == 'tricosine':
        return loss_tricosine
    elif orientation == 'alpha':
        return loss_alpha
    elif orientation == 'rot_y':
        return loss_rot_y
    elif orientation == 'multibin':
        return loss_multibin
    else:
        raise Exception('Incorrect orientation type for loss function')
        

if __name__ == "__main__":
    BATCH_SIZE = args.batch_size
    ORIENTATION = args.orientation
    KITTI_DIR = args.kitti_dir
    WEIGHT = args.weight_path
    WORKERS = args.workers
    OUTPUT_DIR = args.output_dir
    if not os.path.isdir(KITTI_DIR):
        raise Exception('kitti_dir is not a directory.')
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    LABEL_DIR = os.path.join(KITTI_DIR, 'training/label_2/')
    IMG_DIR = os.path.join(KITTI_DIR, 'training/image_2/')

    if ORIENTATION=='alpha':
        PREDICTION_TARGET = 'alpha'
    else:
        PREDICTION_TARGET = 'rot_y'
    # Generator config
    test_gen = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                  orientation_type=ORIENTATION, mode='test',prediction_target=PREDICTION_TARGET)
    print('Testing on {:n} objects. '.format(len(test_gen.obj_ids)))
    # Building Model
    inputs = Input(shape=(224, 224, 3))
    x = Xception_model(inputs, pooling='avg')
    x = add_output_layers(ORIENTATION, x)
    model = Model(inputs=inputs, outputs=x)
    loss_func, loss_weights = get_loss_params(ORIENTATION)
    model.compile(loss=loss_func, optimizer='adam',
                  metrics=[OrientationAccuracy(ORIENTATION)], run_eagerly=True)
    model.load_weights(WEIGHT)
    start_time = time.time()

    predictions = model.predict(x=test_gen,verbose=1,workers= WORKERS,use_multiprocessing=False) # this speeds up the code speed by 4x, but is too hard to work with, will recommend multiprocessing false for training
    file_output = []

    for i, pred in enumerate(predictions):
        preds = {}
        if ORIENTATION=='alpha':
            pred_alpha = conv.angle_normed_to_radians(pred[0])
            preds['norm_alpha'] = pred
            preds['pred_alpha'] = pred_alpha
        elif ORIENTATION == 'rot_y':
            pred_roty = conv.angle_normed_to_radians(pred[0])
            tokens = test_gen.all_objs[i]['line'].strip().split(' ')
            preds['norm_roty'] = pred
            preds['pred_roty'] = pred_roty
            preds['conv_roty'] = conv.rot_y_to_alpha(pred_roty,float(tokens[11]),float(tokens[13]))
        elif ORIENTATION == 'single_bin':
            conv_single = conv.single_bin_to_radians(pred)
            preds['pred_single'] = pred
            preds['conv_single'] = conv_single
        elif ORIENTATION == 'voting_bin':
            conv_voting = conv.voting_bin_to_radians(pred)
            preds['voting_pred'] = pred
            preds['conv_voting'] = conv_voting
        elif ORIENTATION =='tricosine':
            conv_tri = conv.tricosine_to_radians(pred)
            preds['tricosine_pred'] = pred
            preds['conv_tricosine'] = conv_tri
        file_output.append({'pred':preds, # pred outputs in orientation type
                       'line':test_gen.all_objs[i]['line'], # kitti line
                       'img_id':test_gen.all_objs[i]['image_file'][0:6]})
    with open(os.path.join("preds", ORIENTATION+".json"), "w") as fp:
        json.dump(file_output, fp, cls=NumpyEncoder)
    
    
