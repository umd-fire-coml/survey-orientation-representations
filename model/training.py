from logging import log
from numpy import dtype
from numpy.core.fromnumeric import sort
import tensorflow as tf
from build_model import build_model
from loss_function import get_loss_params
from metrics import OrientationAccuracy
import data_processing as dp
import os, re, argparse, time, sys
from datetime import datetime
import pathlib
import tensorflow as tf
import orientation_converters
from orientation_converters import get_output_shape_dict as output_shape
sys.path.append('../')
import utils.train_utils as train_utils


# setup and config gpu
train_utils.setup_gpu()
# Processing argument
parser = argparse.ArgumentParser(description='Training Model')
args = train_utils.setup_cmd_arg(parser)
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
    train_utils.check_args(args, DEPTH_PATH_DIR,LABEL_DIR, IMG_DIR)
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

    # Generator config
    train_gen = dp.KittiGenerator(
        label_dir=LABEL_DIR,
        image_dir=IMG_DIR,
        batch_size=BATCH_SIZE,
        orientation_type=ORIENTATION,
        mode='train',
        val_split=VAL_SPLIT,
        prediction_target=PREDICTION_TARGET,
        add_pos_enc=ADD_POS_ENC,
        add_depth_map=ADD_DEPTH_MAP,
    )
    val_gen = dp.KittiGenerator(
        label_dir=LABEL_DIR,
        image_dir=IMG_DIR,
        batch_size=BATCH_SIZE,
        orientation_type=ORIENTATION,
        mode='val',
        val_split=VAL_SPLIT,
        all_objs=train_gen.all_objs,
        prediction_target=PREDICTION_TARGET,
        add_pos_enc=ADD_POS_ENC,
        add_depth_map=ADD_DEPTH_MAP,
    )
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
        run_eagerly=True
    )

    # early stop callback and accuracy callback
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=20)
    if RESUME:
        latest_training_dir, latest_epoch, latest_weight = train_utils.find_latest_epoch_and_weights(weights_directory, verbose = True)
        init_epoch = int(latest_epoch)
        if not init_epoch:
            raise Exception("Fail to match epoch number")
        if init_epoch == 1:
            raise Exception("No existing record found!")
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

    train_history = model.fit(
        x=train_gen,
        epochs=NUM_EPOCH,
        verbose=1,
        validation_data=val_gen, 
        callbacks=[tb_callback, cp_callback],
        initial_epoch=init_epoch,
        use_multiprocessing = True,
        workers = 8
    )

    print('Training Finished. Weights and history are saved under directory:', WEIGHT_DIR_ROOT)
    print('Total training time is', train_utils.timer(start_time, time.time()))
