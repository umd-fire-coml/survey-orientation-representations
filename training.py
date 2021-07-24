import tensorflow as tf
from build_model import build_model
from loss_function import get_loss_params
from metrics import OrientationAccuracy
import data_processing as dp
import os
import argparse
import time
from datetime import datetime
import pathlib

import tensorflow as tf

# set up tensorflow GPU
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

# Processing argument
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--predict', dest='predict', type=str, default="rot-y",
                    help='The target angle to be predicted. Options are rot-y, alpha')
parser.add_argument('--converter', dest='orientation', type=str,
                    help='Orientation conversion type of the model. '
                         'Options are alpha, rot_y, tricosine, multibin, voting_bin, single_bin')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                    help='Define the batch size for training. Default value is 8')
parser.add_argument('--epoch', dest='num_epoch', type=int, default=100,
                    help='Number of epoch used for training. Default value is 100')
parser.add_argument('--kitti_dir', dest='kitti_dir', type=str, default='dataset',
                    help='path to kitti dataset directory. Its subdirectory should have training/ and testing/. '
                         'Default path is dataset/')
parser.add_argument('--training_record', dest= 'training_record', type= str, default='training_record',
                    help='root directory of all training record, parent of weights and logs directory. '
                         'Default path is training_record')
parser.add_argument('--log_dir', dest='log_dir', type=str, default = 'logs',
                    help='path to tensorboard logs directory. Default path is training_record/logs')
parser.add_argument('--weight_dir', dest='weight_dir', type=str, default='weights',
                    help='Relative path to save weights. Default path is training_record/weights')
parser.add_argument('--val_split', dest='val_split', type=float, default=0.2,
                    help='Fraction of the dataset used for validation. Default val_split is 0.2')
parser.add_argument('--resume', dest = 'resume', type=bool, default=False)
parser.add_argument('--add_pos_enc', dest = 'add_pos_enc', type=bool, default=False)
args = parser.parse_args()


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":
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
    TRAINING_RECORD = args.training_record

    if not os.path.isdir(KITTI_DIR):
        raise Exception('kitti_dir is not a directory.')
    if ORIENTATION not in ['tricosine', 'alpha', 'rot-y', 'multibin', 'voting_bin', 'single_bin']:
        raise Exception('Invalid Orientation Type.')
    if not 0.0 <= VAL_SPLIT <= 1.0:
        raise Exception('Invalid val_split range between [0.0, 1.0]')
    if not os.path.isdir(WEIGHT_DIR_ROOT):
        os.mkdir(WEIGHT_DIR_ROOT)

    # get training starting time and construct stamps
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + str(int(start_time))
    pos_enc_stamp = "_with_pos_enc" if ADD_POS_ENC else ""
    training_stamp = f'{PREDICTION_TARGET}_{ORIENTATION}_{pos_enc_stamp}'
    training_stamp_with_timestamp = training_stamp + '_' + timestamp
    print(f'training stamp ={training_stamp}\ntraining stamp with timestamp ={training_stamp_with_timestamp}')
    # format for .h5 weight file
    weight_format = 'epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.h5'

    if not RESUME:
        log_dir = os.path.join(LOG_DIR_ROOT, training_stamp_with_timestamp)
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = os.path.join(WEIGHT_DIR_ROOT, training_stamp_with_timestamp)
        pathlib.Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # model callback config
        checkpoint_file_name = os.path.join(checkpoint_dir,weight_format)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_name, save_weights_only=True, verbose=1)
        # tensorboard logs path
        tb_log_dir = os.path.join(log_dir, "logs/scalars/")
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir, histogram_freq=1)

    LABEL_DIR = os.path.join(KITTI_DIR, 'training/label_2/')
    IMG_DIR = os.path.join(KITTI_DIR, 'training/image_2/')

    # Generator config
    train_gen = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                  orientation_type=ORIENTATION, mode='train', val_split=VAL_SPLIT, prediction_target=PREDICTION_TARGET,
                                  add_pos_enc=ADD_POS_ENC)
    val_gen = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                   orientation_type=ORIENTATION, mode='val', val_split=VAL_SPLIT,
                                   all_objs=train_gen.all_objs, prediction_target=PREDICTION_TARGET,
                                   add_pos_enc=ADD_POS_ENC)
    print('Training on {:n} objects. Validating on {:n} objects.'.format(len(train_gen.obj_ids), len(val_gen.obj_ids)))

    # Building Model
    model = build_model(ORIENTATION, ADD_POS_ENC)

    loss_func, loss_weights = get_loss_params(ORIENTATION)

    model.compile(loss=loss_func, loss_weights=loss_weights, optimizer='adam',
                  metrics=OrientationAccuracy(ORIENTATION), run_eagerly=True)


    # early stop callback and accuracy callback
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=20)
    if RESUME:
        old_time_stamp = "tricosine_with_pos_enc_2021-07-21-11-01-05_1626879665"
        weights_file_name = "pred_type-alphaepoch-64-loss-0.0163-val_loss-0.1166.h5"
        init_epoch = 64
        print(f'resuming from {old_time_stamp}\\{weights_file_name}')
        stored_weights_file = os.path.join(WEIGHT_DIR_ROOT, old_time_stamp, weights_file_name)
        if not os.path.isfile(stored_weights_file):
            raise (f'stored weights directory "{stored_weights_file}" is not a valid file')
        model.load_weights(stored_weights_file)
        # overwrite tensorboard callback
        tb_log_dir = os.path.join(WEIGHT_DIR_ROOT, "logs", old_time_stamp, "logs/scalars/")
        if not os.path.isdir(tb_log_dir):
            raise (f'tensorboard log directory "{tb_log_dir}" is not a valid directory')
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
        # overwrite call back directory
        cp_callback_file = os.path.join(WEIGHT_DIR_ROOT, old_time_stamp, weight_format)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_callback_file, save_weights_only=True, verbose=1)

    train_history = model.fit(x=train_gen, epochs=NUM_EPOCH, verbose=1,
                              validation_data=val_gen, callbacks=[tb_callback, cp_callback])

    print('Training Finished. Weights and history are saved under directory:', WEIGHT_DIR_ROOT)
    print('Total training time is', timer(start_time, time.time()))