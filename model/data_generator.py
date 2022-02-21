#! /usr/bin/env python3
import os
import math
import numpy as np
import pathlib
from skimage import io
from skimage.transform import resize
import pickle
from skimage.util import img_as_float
from skimage.transform import resize
from os.path import join
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from random import random
from positional_encoder import get_2d_pos_enc
from orientation_converters import *
from add_output_layers import *

# constants
CROP_RESIZE_H, CROP_RESIZE_W = 224, 224
IMG_H, IMG_W = 376, 1242
TRAIN_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
KITTI_CLASSES = [
    'Cyclist',
    'Tram',
    'Person_sitting',
    'Truck',
    'Pedestrian',
    'Van',
    'Car',
    'Misc',
    'DontCare',
]
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
def get_all_objs_from_kitti_dir(label_dir:str, image_dir:str, difficulty='hard'):
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
            truncated = float(
                obj_line_tokens[1]
            )  # Float from 0 (non-truncated) to 1 (truncated)
            occluded = int(obj_line_tokens[2])  # 0 = fully visible, 1 = partly occluded,
            ymin = int(float(obj_line_tokens[5]))
            ymax = int(float(obj_line_tokens[7]))
            bbox_height = ymax - ymin

            # filter objs based on TRAIN_CLASSES, MIN_HEIGHT, MAX_OCCLUSION, MAX_TRUNCATION
            if (
                class_name in TRAIN_CLASSES
                and bbox_height > MIN_BBOX_HEIGHT[DIFFICULTY_id]
                and occluded <= MAX_OCCLUSION[DIFFICULTY_id]
                and truncated <= MAX_TRUNCATION[DIFFICULTY_id]
            ):

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
                    'line': obj_line,
                }

                # Get camera view angle of the object
                center = ((obj['xmin'] + obj['xmax']) / 2, (obj['ymin'] + obj['ymax']) / 2)
                obj['view_angle'] = center[0] / CROP_RESIZE_W * VIEW_ANGLE_TOTAL_X - (
                    VIEW_ANGLE_TOTAL_X / 2
                )

                # calculate the moving average of each obj dims.
                # accumulate the sum of each dims for each obj
                # get the count of the obj, then times the current avg of dims, + current obj's dim
                class_dims_means[obj['class_name']] = (
                    class_counts[obj['class_name']] * class_dims_means[obj['class_name']]
                    + obj['dims']
                )
                class_counts[obj['class_name']] += 1
                # get the new average
                class_dims_means[obj['class_name']] /= class_counts[obj['class_name']]

                all_objs.append(obj)
    # I have now accumulated all objects into all_objs from kitti data in obj dict format
    return all_objs


# get the bounding box,  values for the instance
# this automatically does flips
# per image
def prepare_generator_output(
    image_dir: str,
    obj,
    orientation_type: str,
    prediction_target: str,
    add_pos_enc: bool,
    add_depth_map: bool,
):
    # Prepare image patch
    xmin = obj['xmin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymin = obj['ymin']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    xmax = obj['xmax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)
    ymax = obj['ymax']  # + np.random.randint(-MAX_JIT, MAX_JIT+1)

    # read object image
    img = img_as_float(io.imread(join(image_dir, obj['image_file'])))

    if add_pos_enc and add_depth_map:
        # prepare pos enc
        pos_enc = get_2d_pos_enc(*img.shape)
        # prepare depth map
        depth_map_path = (
            pathlib.Path(image_dir).parents[0] / "predict_depth" / f'depth_{obj["image_file"]}'
        )
        depth_map = img_as_float(io.imread(depth_map_path))
        depth_map = resize(depth_map, img.shape[:2])
        depth_map = np.expand_dims(depth_map, -1)
        # concatenate together
        stacked = np.concatenate((img, pos_enc, depth_map), axis=-1)
        # crop the image
        img = stacked[ymin : ymax + 1, xmin : xmax + 1]
    elif add_pos_enc:
        pos_enc = get_2d_pos_enc(*img.shape)
        stacked = np.concatenate((img, pos_enc), axis=-1)
        img = stacked[ymin : ymax + 1, xmin : xmax + 1]
    elif add_depth_map:
        depth_map_path = (
            pathlib.Path(image_dir).parents[0] / "predict_depth" / f'depth_{obj["image_file"]}'
        )
        depth_map = img_as_float(io.imread(depth_map_path))
        depth_map = resize(depth_map, img.shape[:2])
        depth_map = np.expand_dims(depth_map, -1)
        stacked = np.concatenate((img, depth_map), axis=-1)
        img = stacked[ymin : ymax + 1, xmin : xmax + 1]

    else:
        img = img[ymin : ymax + 1, xmin : xmax + 1]

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
                # multibin_orientation_flipped, multibin_confidence_flipped = radians_to_multibin(math.tau - obj[prediction_target])
                (
                    multibin_orientation_flipped,
                    multibin_confidence_flipped,
                ) = radians_to_multi_affinity_bin(math.tau - obj[prediction_target])
                obj['multibin_orientation_flipped'] = multibin_orientation_flipped
                obj['multibin_confidence_flipped'] = multibin_confidence_flipped
            return img, np.concatenate(
                (obj['multibin_orientation_flipped'], obj['multibin_confidence_flipped']),
                axis=-1,
            )
        
        elif orientation_type == 'tricosine':
            if 'tricosine_flipped' not in obj:
                obj['tricosine_flipped'] = radians_to_tricosine(
                    math.tau - obj[prediction_target]
                )
            return img, obj['tricosine_flipped']
        
        elif orientation_type == 'voting-bin':
            if 'voting_bin_flipped' not in obj:
                obj['voting-bin_flipped'] = radians_to_voting_bin(
                    math.tau - obj[prediction_target]
                )
            return img, obj['voting-bin_flipped']
        
        elif orientation_type == 'single-bin':
            if 'single-bin_flipped' not in obj:
                obj['single-bin_flipped'] = radians_to_single_bin(
                    math.tau - obj[prediction_target]
                )
            return img, obj['single-bin_flipped']
        
        elif orientation_type == 'alpha' and prediction_target == 'alpha':
            if 'alpha_normed_flipped' not in obj:
                obj['alpha_normed_flipped'] = radians_to_angle_normed(math.tau - obj['alpha'])
            return img, obj['alpha_normed_flipped']
        
        elif orientation_type == 'rot-y' and prediction_target == 'rot-y':
            if 'rot-y_normed_flipped' not in obj:
                obj['rot-y_normed_flipped'] = radians_to_angle_normed(math.tau - obj['rot-y'])
            return img, obj['rot-y_normed_flipped']
        
        elif orientation_type == 'exp-A':
            if 'exp-A_flipped' not in obj:
                obj['exp-A_flipped'] = radians_to_expA(
                    math.tau - obj[prediction_target]
                )
            return img, obj['exp-A_flipped']
        else:
            raise Exception(
                f"Invalid orientation_type: {orientation_type}, with prediction_target:"
                f" {prediction_target}"
            )
    else:
        if orientation_type == 'multibin':
            if 'multibin_orientation' not in obj:
                # Get orientation and confidence values for flip
                # multibin_orientation, multibin_confidence = radians_to_multibin(obj[prediction_target])
                multibin_orientation, multibin_confidence = radians_to_multi_affinity_bin(
                    obj[prediction_target]
                )
                obj['multibin_orientation'] = multibin_orientation
                obj['multibin_confidence'] = multibin_confidence
            return img, np.concatenate(
                (obj['multibin_orientation'], obj['multibin_confidence']), axis=-1
            )
        
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
        
        elif orientation_type == 'exp-A':
            if 'exp-A' not in obj:
                obj['exp-A'] = radians_to_expA(
                    math.tau - obj[prediction_target]
                )
            return img, obj['exp-A']
        
        else:
            raise Exception(
                f"Invalid orientation_type: {orientation_type}, with prediction_target:"
                f" {prediction_target}"
            )


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

    def __init__(
        self,
        # when using tf.data.dataset.from_generator, it convert all str argument to binary
        label_dir: bytes = b'dataset/training/label_2/',
        image_dir: bytes = b'dataset/training/image_2/',
        mode: str = "train",
        get_kitti_line: bool = False,
        orientation_type: str = "multibin",
        val_split: float = 0.0,
        prediction_target: str = 'rot-y',
        add_pos_enc: bool = False,
        add_depth_map: bool = False,
    ):
        self.label_dir = label_dir.decode() if isinstance(label_dir,bytes) else label_dir
        self.image_dir = image_dir.decode() if isinstance(image_dir,bytes) else image_dir
        self.get_kitti_line = get_kitti_line
        self.mode = mode.decode() if isinstance(mode,bytes) else mode
        self.orientation_type = orientation_type.decode() if isinstance(orientation_type,bytes) else orientation_type
        self.prediction_target = prediction_target.decode() if isinstance(prediction_target,bytes) else prediction_target
        self.add_pos_enc = add_pos_enc
        self.add_depth_map = add_depth_map
        # load all kitti objects and save loaded data
        pkl_file = pathlib.Path(f"tmp_data.pkl")
        if pkl_file.is_file():
            with open(pkl_file,'rb') as handle:
                self.all_objs = pickle.load(handle)
        else:
            self.all_objs = get_all_objs_from_kitti_dir(label_dir.decode(), image_dir.decode())
            print(f'pickle file {pkl_file} not found, saving it locally')
            with open(pkl_file,'wb') as handle:
                pickle.dump(self.all_objs, handle)

        
        self.obj_ids = list(
            range(len(self.all_objs))
        )  # list of all object indexes for the generator

        if val_split > 0.0:
            assert mode != 'all' and val_split < 1.0
            cutoff = int(val_split * len(self.all_objs))
            if self.mode == "train":
                self.obj_ids = self.obj_ids[cutoff:]
            elif self.mode == "val":
                self.obj_ids = self.obj_ids[:cutoff]  # reduce range for testing
            else:
                raise Exception(f"invalid mode: {self.mode}")
        

    def __len__(self):
        return len(self.obj_ids)

    def __getitem__(self, idx):
        # prepare batch of images
        n_channel = 3  # by defualt 3 channels: RGB
        if self.add_depth_map and self.add_pos_enc:
            n_channel = 7  # RGB + positional encodin + depth map
        elif self.add_pos_enc:
            n_channel = 6  # RGB + positional encoding
        elif self.add_depth_map:
            n_channel = 4  # RGB + depth map

        # prepare kitti line output for visualization
        line_batch = []
        # insert data
        img, orientation = prepare_generator_output(
            self.image_dir,
            self.all_objs[idx],
            self.orientation_type,
            self.prediction_target,
            self.add_pos_enc,
            self.add_depth_map,
        )
        if self.get_kitti_line:
            line_batch.append(self.all_objs[idx]['line'])
        label = orientation
        if self.get_kitti_line:
            label['line_batch'] = line_batch
        return img, label

    def on_epoch_end(self):
        np.random.shuffle(self.obj_ids)

    def __str__(self):
        return (
            "KittiDatagenerator:<size %d, orientation_type: %s, image_dir:%s, label_dir:%s,"
            " epoch:%d>"
            % (len(self), self.orientation_type, self.image_dir, self.label_dir, self.epochs)
        )