#! /usr/bin/env python3
import os
import math
import numpy as np
import copy
from skimage import io
from skimage.util import img_as_float
from skimage.transform import resize
from os.path import join
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from random import random
from positional_encoder import get_2d_pos_enc
from orientation_converters import (radians_to_angle_normed, radians_to_multibin, radians_to_single_bin, radians_to_tricosine, radians_to_voting_bin,
SHAPE_ALPHA_ROT_Y, SHAPE_MULTIBIN, SHAPE_SINGLE_BIN, SHAPE_TRICOSINE, SHAPE_VOTING_BIN)
from add_output_layers import LAYER_OUTPUT_NAME_MULTIBIN, LAYER_OUTPUT_NAME_TRICOSINE, LAYER_OUTPUT_NAME_ALPHA_ROT_Y, LAYER_OUTPUT_NAME_VOTING_BIN, LAYER_OUTPUT_NAME_SINGLE_BIN

# constants
CROP_RESIZE_H, CROP_RESIZE_W = 224, 224
IMG_H, IMG_W = 376, 1242
TRAIN_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
KITTI_CLASSES = ['Cyclist', 'Tram', 'Person_sitting', 'Truck', 'Pedestrian', 'Van', 'Car', 'Misc', 'DontCare']
DIFFICULTY = ['easy', 'moderate', 'hard']
MIN_BBOX_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
NUMPY_TYPE = np.float32
VIEW_ANGLE_TOTAL_X = 1.4835298642
VIEW_ANGLE_TOTAL_Y = 0.55850536064

# store the average dimensions of different train classes
class_dims_means = {key: np.asarray([0, 0, 0]) for key in TRAIN_CLASSES}
# store the count of different train classes
class_counts = {key: 0 for key in TRAIN_CLASSES}

# this creates a list of dict for each obj from from the kitti train val directories
def get_all_objs_from_kitti_dir(label_dir, image_dir, difficulty='hard'):
    # get difficulty index
    DIFFICULTY_id = DIFFICULTY.index(difficulty)
    # store all objects
    all_objs = []

    for image_file in tqdm(os.listdir(image_dir)):
        label_file = image_file.replace('.png', '.txt')

        # each line represents an object
        for obj_line in open(join(label_dir, label_file)).readlines():
            obj_line_tokens = obj_line.strip().split(' ')
            class_name = obj_line_tokens[0]
            truncated = float(obj_line_tokens[1]) # Float from 0 (non-truncated) to 1 (truncated)
            occluded = int(obj_line_tokens[2]) # 0 = fully visible, 1 = partly occluded,
            ymin = int(float(obj_line_tokens[5]))
            ymax = int(float(obj_line_tokens[7]))
            bbox_height = ymax - ymin

            # filter objs based on TRAIN_CLASSES, MIN_HEIGHT, MAX_OCCLUSION, MAX_TRUNCATION
            if (class_name in TRAIN_CLASSES 
                and bbox_height > MIN_BBOX_HEIGHT[DIFFICULTY_id] 
                and occluded <= MAX_OCCLUSION[DIFFICULTY_id]
                and truncated <= MAX_TRUNCATION[DIFFICULTY_id]):

                obj = {
                        'image_file': image_file,
                        'class_name': class_name,
                        'class_id': KITTI_CLASSES.index(class_name),
                        'truncated': truncated,
                        'occluded': occluded,
                        'alpha': float(obj_line_tokens[3]),
                        'xmin': int(float(obj_line_tokens[4])),
                        'ymin': ymin,
                        'xmax': int(float(obj_line_tokens[6])),
                        'ymax': ymax,
                        'height': float(obj_line_tokens[8]),
                        'width': float(obj_line_tokens[9]),
                        'length': float(obj_line_tokens[10]),
                        'dims': np.asarray([float(number) for number in obj_line_tokens[8:11]]),
                        'loc_x': float(obj_line_tokens[11]),
                        'loc_y': float(obj_line_tokens[12]),
                        'loc_z': float(obj_line_tokens[13]),
                        'rot-y': float(obj_line_tokens[14]),
                        'line': obj_line
                       }

                # Get camera view angle of the object
                center = ((obj['xmin'] + obj['xmax']) / 2, (obj['ymin'] + obj['ymax']) / 2)
                obj['view_angle'] = center[0] / CROP_RESIZE_W * VIEW_ANGLE_TOTAL_X - (VIEW_ANGLE_TOTAL_X / 2)

                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                class_dims_means[obj['class_name']] = class_counts[obj['class_name']] * \
                                              class_dims_means[obj['class_name']] + obj['dims']
                class_counts[obj['class_name']] += 1
                # get the new average
                class_dims_means[obj['class_name']] /= class_counts[obj['class_name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data in obj dict format
    return all_objs

# get the bounding box,  values for the instance
# this automatically does flips
# per image
def prepare_generator_output(image_dir: str, obj, orientation_type: str, prediction_target: str, add_pos_enc: bool):
    # Prepare image patch
    xmin = obj['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = obj['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = obj['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = obj['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

    # read object image
    img = img_as_float(io.imread(join(image_dir, obj['image_file'])))

    if add_pos_enc:
        pos_enc = get_2d_pos_enc(*img.shape)
        stacked = np.concatenate((img, pos_enc), axis=-1)
        img = stacked[ymin:ymax + 1, xmin:xmax + 1]
    else:
        img = img[ymin:ymax + 1, xmin:xmax + 1]
        
    # resize the image crop to standard size
    img = resize(img, (CROP_RESIZE_H, CROP_RESIZE_W), anti_aliasing=True)
    img = img.astype(NUMPY_TYPE)

    # Get the dimensions offset from average (basically zero centering the values)
    obj['dims_mean_offset'] = obj['dims'] - class_dims_means[obj['class_name']]
 
    # flip the image by random chance
    flip = random() < 0.5

    # flip image horizontally
    if flip:
        img = np.fliplr(img)
        if orientation_type == 'multibin':
            if 'multibin_orientation_flipped' not in obj:
                # Get orientation and confidence values for flip
                # multibin_orientation_flipped, multibin_confidence_flipped = alpha_to_multibin_orientation_confidence(math.tau - obj["alpha"])
                multibin_orientation_flipped, multibin_confidence_flipped = radians_to_multibin(math.tau - obj[prediction_target])
                obj['multibin_orientation_flipped'] = multibin_orientation_flipped
                obj['multibin_confidence_flipped'] = multibin_confidence_flipped
            return img, np.concatenate((obj['multibin_orientation_flipped'], obj['multibin_confidence_flipped']), axis=-1)
        elif orientation_type == 'tricosine':
            if 'tricosine_flipped' not in obj:
                obj['tricosine_flipped'] = radians_to_tricosine(math.tau - obj[prediction_target])
            return img, obj['tricosine_flipped']
        elif orientation_type == 'voting-bin':
            if 'voting_bin_flipped' not in obj:
                obj['voting-bin_flipped'] = radians_to_voting_bin(math.tau - obj[prediction_target])
            return img, obj['voting-bin_flipped']
        elif orientation_type == 'single-bin':
            if 'single-bin_flipped' not in obj:
                obj['single-bin_flipped'] = radians_to_single_bin(math.tau - obj[prediction_target])
            return img, obj['single-bin_flipped']
        elif orientation_type == 'alpha' and prediction_target == 'alpha':
            if 'alpha_normed_flipped' not in obj:
                obj['alpha_normed_flipped'] = radians_to_angle_normed(math.tau - obj['alpha'])
            return img, obj['alpha_normed_flipped']  
        elif orientation_type == 'rot-y' and prediction_target == 'rot-y':
            if 'rot-y_normed_flipped' not in obj:
                obj['rot-y_normed_flipped'] = radians_to_angle_normed(math.tau - obj['rot_y'])
            return img, obj['rot-y_normed_flipped']
        else:
            raise Exception(f"Invalid orientation_type: {orientation_type}, with prediction_target: {prediction_target}")
    else:
        if orientation_type == 'multibin':
            if 'multibin_orientation' not in obj:
                # Get orientation and confidence values for flip
                # multibin_orientation, multibin_confidence = alpha_to_multibin_orientation_confidence(obj["alpha"])
                multibin_orientation, multibin_confidence = radians_to_multibin(obj[prediction_target])
                obj['multibin_orientation'] = multibin_orientation
                obj['multibin_confidence'] = multibin_confidence
            return img, np.concatenate((obj['multibin_orientation'], obj['multibin_confidence']), axis=-1)
        elif orientation_type == 'tricosine':
            if 'tricosine' not in obj:
                obj['tricosine'] = radians_to_tricosine(obj[prediction_target])
            return img, obj['tricosine']
        elif orientation_type == 'voting-bin':
            if 'voting-bin' not in obj:
                obj['voting-bin'] = radians_to_voting_bin(obj[prediction_target])
            return img, obj['voting-bin']
        elif orientation_type == 'single-bin':
            if 'single-bin' not in obj:
                obj['single-bin'] = radians_to_single_bin(obj[prediction_target])
            return img, obj['single-bin']
        elif orientation_type == 'alpha' and prediction_target == 'alpha':
            if 'alpha_normed' not in obj:
                obj['alpha_normed'] = radians_to_angle_normed(obj['alpha'])
            return img, obj['alpha_normed']
        elif orientation_type == 'rot-y' and prediction_target == 'rot-y':
            if 'rot-y_normed' not in obj:
                obj['rot-y_normed'] = radians_to_angle_normed(obj['rot-y'])
            return img, obj['rot-y_normed']
        else:
            raise Exception(f"Invalid orientation_type: {orientation_type}, with prediction_target: {prediction_target}")

            
class KittiGenerator(Sequence):
    '''Creates A KittiGenerator Sequence
    Args:
        label_dir (str) : path to the directory with labels
        image_dir (str) : path to the image directory
        mode (str): tells whether to be in train, viz, or all mode
        batch_size (int) : tells batchsize to use
        orientation_type (str): type of oridentation multibin, tricosine, alpha, or rot_y
        val_split (float): what percentage data reserved for validation
    '''

    def __init__(self, label_dir: str = 'dataset/training/label_2/',
                 image_dir: str = 'dataset/training/image_2/',
                 mode: str = "train",
                 get_kitti_line: bool = False,
                 batch_size: int = 8,
                 orientation_type: str = "multibin",
                 val_split: float = 0.0,
                 prediction_target: str = 'rot_y',
                 all_objs = None,
                 add_pos_enc: bool = False):
        self.label_dir = label_dir
        self.image_dir = image_dir
        if all_objs == None:
            self.all_objs = get_all_objs_from_kitti_dir(label_dir, image_dir)
        else:
            self.all_objs = all_objs
        self.get_kitti_line = get_kitti_line
        self.mode = mode
        self.batch_size = batch_size
        self.orientation_type = orientation_type
        self.prediction_target = prediction_target
        self.obj_ids = list(range(len(self.all_objs)))  # list of all object indexes for the generator
        self.add_pos_enc = add_pos_enc
        if val_split > 0.0:
            assert mode != 'all' and val_split < 1.0
            cutoff = int(val_split * len(self.all_objs))  
            if self.mode == "train":
                self.obj_ids = self.obj_ids[cutoff:]
            elif self.mode == "val":
                self.obj_ids = self.obj_ids[:cutoff] # reduce range for testing
            else:
                raise Exception("invalid mode")
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.obj_ids) // self.batch_size)

    def __getitem__(self, idx):
        l_bound = idx * self.batch_size  # start of key index
        r_bound = l_bound + self.batch_size  # end of key index
        r_bound = r_bound if r_bound < len(self.obj_ids) else len(self.obj_ids)  # check for key index overflow
        num_batch_objs = r_bound - l_bound

        # prepare batch of images
        n_channel = 6 if self.add_pos_enc else 3
        img_batch = np.empty((num_batch_objs, CROP_RESIZE_H, CROP_RESIZE_W, n_channel))

        # prepare batch of orientation_type tensor
        if self.orientation_type == "multibin":
            orientation_batch = np.empty((num_batch_objs, *SHAPE_MULTIBIN))
        elif self.orientation_type == 'tricosine':
            orientation_batch = np.empty((num_batch_objs, *SHAPE_TRICOSINE))
        elif self.orientation_type == "alpha" or self.orientation_type == 'rot_y':
            orientation_batch = np.empty((num_batch_objs, *SHAPE_ALPHA_ROT_Y))
        elif self.orientation_type == "voting-bin":
            orientation_batch = np.empty((num_batch_objs, *SHAPE_VOTING_BIN))
        elif self.orientation_type == "single-bin":
            orientation_batch = np.empty((num_batch_objs, *SHAPE_SINGLE_BIN))
        else:
            raise Exception("Invalid Orientation Type")

        # prepare kitti line output for visualization
        line_batch = []

        # insert data
        for i, obj_id in enumerate(self.obj_ids[l_bound:r_bound]):
            img, orientation = prepare_generator_output(self.image_dir,
                                                        self.all_objs[obj_id],
                                                        self.orientation_type,
                                                        self.prediction_target,
                                                        self.add_pos_enc)
            img_batch[i] = img
            orientation_batch[i] = orientation
            if self.get_kitti_line:
                line_batch.append(self.all_objs[obj_id]['line'])
 
        if self.orientation_type == "multibin":
            y_batch = {LAYER_OUTPUT_NAME_MULTIBIN: orientation_batch}
        elif self.orientation_type == 'tricosine':
            y_batch = {LAYER_OUTPUT_NAME_TRICOSINE: orientation_batch}
        elif self.orientation_type == "alpha" or self.orientation_type == 'rot_y':
            y_batch = {LAYER_OUTPUT_NAME_ALPHA_ROT_Y: orientation_batch}
        elif self.orientation_type == "voting-bin":
            y_batch = {LAYER_OUTPUT_NAME_VOTING_BIN: orientation_batch}
        elif self.orientation_type == "single-bin":
            y_batch = {LAYER_OUTPUT_NAME_SINGLE_BIN: orientation_batch}
        else:
            raise Exception("Invalid Orientation Type")
        
        if self.get_kitti_line:
            y_batch['line_batch'] = line_batch

        return img_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.obj_ids)

    def __str__(self):
        return "KittiDatagenerator:<size %d, orientation_type: %s, image_dir:%s, label_dir:%s, epoch:%d>" % (
        len(self), self.orientation_type, self.image_dir, self.label_dir, self.epochs)
