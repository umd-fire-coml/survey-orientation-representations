from logging import log
from numpy import dtype
from numpy.core.fromnumeric import sort
import tensorflow as tf
from build_model import build_model
from loss_function import get_loss_params
from metrics import OrientationAccuracy
import data_generator as dp
import os, re, argparse, time, sys
from datetime import datetime
import pathlib
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.data import AUTOTUNE
import orientation_converters
from orientation_converters import get_output_shape_dict as output_shape
sys.path.append('../')
from utils.train_utils import *

DATASET_TOTAL_OBJECTS = 26937
# setup and config gpu
setup_gpu()
# Processing argument
parser = argparse.ArgumentParser(description='Training Model')
args = setup_cmd_arg(parser)
BATCH_SIZE = args.batch_size
NUM_EPOCH = args.num_epoch
ORIENTATION = args.orientation
KITTI_DIR = args.kitti_dir
WEIGHT_DIR_ROOT = args.weight_dir
LOG_DIR_ROOT = args.log_dir
VAL_SPLIT = args.val_split
PREDICTION_TARGET = args.predict
RESUME = args.resume
ADD_POS_ENC = args.add_pos_enc
TRAINING_RECORD = pathlib.Path(args.training_record)
ANGULAR_LOSS = args.use_angular_loss
ADD_DEPTH_MAP = args.add_depth_map
DEPTH_PATH_DIR = os.path.join(KITTI_DIR, "training/predict_depth")
LABEL_DIR = os.path.join(KITTI_DIR, 'training/label_2/')
IMG_DIR = os.path.join(KITTI_DIR, 'training/image_2/')


if __name__ == "__main__":
    # checking if receving valid arguments
    check_args(args, DEPTH_PATH_DIR,LABEL_DIR, IMG_DIR)
    # get training starting time and construct stamps
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + str(int(start_time))
    training_stamp = f'{PREDICTION_TARGET}_{ORIENTATION}'
    if ADD_POS_ENC:
        training_stamp += "_with_pos_enc"
    if ADD_DEPTH_MAP:
        training_stamp += "_with_depth_map"
    training_stamp += f"_{timestamp}"
    print(f'training stamp with timestamp:{training_stamp}')
    # format for .h5 weight file
    # old weight_format = 'epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.h5'
    weight_format = 'epoch-{epoch:02d}-val_acc-{val_orientation_accuracy:.4f}-train_acc-{orientation_accuracy:.4f}-val_loss-{val_loss:.4f}-train_loss-{loss:.4f}.h5'
    weights_directory = TRAINING_RECORD / 'weights' if not WEIGHT_DIR_ROOT else WEIGHT_DIR_ROOT
    logs_directory = TRAINING_RECORD / 'logs' if not LOG_DIR_ROOT else LOG_DIR_ROOT
    weights_directory.mkdir(parents=True, exist_ok=True)
    logs_directory.mkdir(parents=True, exist_ok=True)
    init_epoch = 0
    
    if not RESUME:
        log_dir = logs_directory / training_stamp
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = weights_directory / training_stamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # model callback config
        checkpoint_file_name = checkpoint_dir / weight_format
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_name, save_weights_only=True, verbose=1
        )
        # tensorboard logs path
        tb_log_dir = log_dir / "logs/scalars/"
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

    train_dataset = tf.data.Dataset.from_generator(
        dp.KittiGenerator,
        output_signature=(
            tf.TensorSpec(shape=(224,224,3), dtype=tf.float32), # image 
            tf.TensorSpec(shape=output_shape()[str(ORIENTATION)], dtype=tf.float32)  # label
            ),
        args=(
             LABEL_DIR,  # label dir
             IMG_DIR,    # image dir
             "train",    # mode
             False,      # get kitti line
             ORIENTATION,# orientation type
             VAL_SPLIT,  # validation split
             PREDICTION_TARGET,  # prediction target
             False,      # add positional encoding
             False,      # add depth map
         )
    )

    val_dataset = tf.data.Dataset.from_generator(
        dp.KittiGenerator,
        output_signature=(
            tf.TensorSpec(shape=(224,224,3), dtype=tf.float32), # image 
            tf.TensorSpec(shape=output_shape()[str(ORIENTATION)], dtype=tf.float32)  # label
            ),
        args=(
                LABEL_DIR,  # label dir
                IMG_DIR,    # image dir
                "val",      # mode
                False,      # get kitti line
                ORIENTATION,# orientation type
                VAL_SPLIT,  # validation split
                PREDICTION_TARGET,  # prediction target
                False,      # add positional encoding
                False,      # add depth map
            )
    )
    train_dataset = (train_dataset
        .prefetch(AUTOTUNE)
        .batch(BATCH_SIZE)
        .shuffle(50)
        .repeat(1)
    )
    val_dataset = (val_dataset
        .prefetch(AUTOTUNE)
        .batch(BATCH_SIZE))

    # Building Model
    n_channel = 3
    if ADD_DEPTH_MAP and ADD_DEPTH_MAP:
        n_channel = 7
    elif ADD_POS_ENC:
        n_channel = 6
    elif ADD_DEPTH_MAP:
        n_channel = 4
    height = dp.CROP_RESIZE_H
    width = dp.CROP_RESIZE_W
    model = build_model(ORIENTATION, height, width, n_channel)

    loss_func, loss_weights = get_loss_params(ORIENTATION, ANGULAR_LOSS)

    model.compile(
        loss=loss_func,
        loss_weights=loss_weights,
        optimizer='adam',
        metrics=OrientationAccuracy(ORIENTATION),
        run_eagerly=True,
    )

    # early stop callback and accuracy callback
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=20)
    if RESUME:
        latest_training_dir, latest_epoch, latest_weight = find_latest_epoch_and_weights(weights_directory, verbose = True)
        init_epoch = int(latest_epoch)
        if not init_epoch:
            raise Exception("Fail to match epoch number")
        if not os.path.isfile(latest_weight):
            raise FileNotFoundError(
                f'stored weights directory "{latest_weight}" is not a valid file'
            )
        model.load_weights(latest_weight)
        # overwrite tensorboard callback
        print(f'current log directory: {logs_directory}')
        tb_log_dir = logs_directory / latest_training_dir.name / "logs" / "scalars"
        if not tb_log_dir.is_dir():
            raise FileNotFoundError(
                f'tensorboard log directory "{tb_log_dir}" is not a valid directory'
            )
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
        # overwrite call back directory
        cp_callback_file = weights_directory / latest_training_dir / weight_format
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_callback_file, save_weights_only=True, verbose=1
        )
    train_total_obj = int((1-VAL_SPLIT) * DATASET_TOTAL_OBJECTS)
    print(f'Total number of objects to train: {train_total_obj} \nStep size: {int(train_total_obj/NUM_EPOCH)}')
    train_history = model.fit(
        x=train_dataset,
        epochs=NUM_EPOCH,
        steps_per_epoch = train_total_obj // BATCH_SIZE,
        verbose=1,
        validation_data=val_dataset, # need to change to validation dataset later
        callbacks=[tb_callback, cp_callback],
        initial_epoch=init_epoch,
        use_multiprocessing = True,
        workers = 8
    )

    print('Training Finished. Weights and history are saved under directory:', WEIGHT_DIR_ROOT)
    print('Total training time is', timer(start_time, time.time()))
