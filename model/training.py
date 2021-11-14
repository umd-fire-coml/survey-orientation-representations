from logging import log
from numpy.core.fromnumeric import sort
import tensorflow as tf
from build_model import build_model
from loss_function import get_loss_params
from metrics import OrientationAccuracy
import data_processing as dp
import os, re, argparse, time, glob
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
                         'Options are alpha, rot-y, tricosine, multibin, voting-bin, single-bin')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                    help='Define the batch size for training. Default value is 8')
parser.add_argument('--epoch', dest='num_epoch', type=int, default=100,
                    help='Number of epoch used for training. Default value is 100')
parser.add_argument('--kitti_dir', dest='kitti_dir', type=str, default='dataset',
                    help='path to kitti dataset directory. Its subdirectory should have training/ and testing/. '
                         'Default path is dataset/')
parser.add_argument('--training_record', dest= 'training_record', type=str, default='training_record',
                    help='root directory of all training record, parent of weights and logs directory. '
                         'Default path is training_record')
parser.add_argument('--log_dir', dest='log_dir', type=str,
                    help='path to tensorboard logs directory. Default path is training_record/logs')
parser.add_argument('--weight_dir', dest='weight_dir', type=str,
                    help='Relative path to save weights. Default path is training_record/weights')
parser.add_argument('--val_split', dest='val_split', type=float, default=0.2,
                    help='Fraction of the dataset used for validation. Default val_split is 0.2')
parser.add_argument('--resume', dest='resume', type=bool, default=False)
parser.add_argument('--add_pos_enc', dest='add_pos_enc', type=bool, default=False)
parser.add_argument("--use_angular_loss", dest='use_angular_loss', type=bool, default= False)
parser.add_argument("--add_depth_map", dest="add_depth_map", type=bool, default=False,
                    help="If add_depth_map is true, add the path to directory containing depth map.")
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
    TRAINING_RECORD = pathlib.Path(args.training_record)   
    ANGULAR_LOSS = args.use_angular_loss
    ADD_DEPTH_MAP = args.add_depth_map

    DEPTH_PATH_DIR = os.path.join(KITTI_DIR, "training/predict_depth")
    LABEL_DIR = os.path.join(KITTI_DIR, 'training/label_2/')
    IMG_DIR = os.path.join(KITTI_DIR, 'training/image_2/')

    if not os.path.isdir(KITTI_DIR):
        raise Exception('kitti_dir is not a directory.')
    if ORIENTATION not in ['tricosine', 'alpha', 'rot-y', 'multibin', 'voting-bin', 'single-bin']:
        raise Exception('Invalid Orientation Type.')
    if not 0.0 <= VAL_SPLIT <= 1.0:
        raise Exception('Invalid val_split range between [0.0, 1.0]')
    if ADD_DEPTH_MAP and (not os.path.isdir(DEPTH_PATH_DIR)):
        raise Exception("Unable to find depth maps. Please put depth map under /kitti_dataset/training/predic_depth")

    # get training starting time and construct stamps
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + str(int(start_time))
    training_stamp = f'{PREDICTION_TARGET}_{ORIENTATION}'
    if ADD_POS_ENC:
        training_stamp += "_with_pos_enc"
    if ADD_DEPTH_MAP:
        training_stamp +="_with_depth_map"
    training_stamp += f"_{timestamp}"
    print(f'training stamp with timestamp:{training_stamp}')
    # format for .h5 weight file
    # old weight_format = 'epoch-{epoch:02d}-loss-{loss:.4f}-val_loss-{val_loss:.4f}.h5'
    weight_format = 'epoch-{epoch:02d}-val_acc-{val_orientation_accuracy:.4f}-train_acc-{orientation_accuracy:.4f}-val_loss-{val_loss:.4f}-train_loss-{loss:.4f}.h5'
    weights_directory = TRAINING_RECORD / 'weights' if not WEIGHT_DIR_ROOT else WEIGHT_DIR_ROOT
    logs_directory = TRAINING_RECORD /  'logs' if not LOG_DIR_ROOT else LOG_DIR_ROOT
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
            filepath=checkpoint_file_name, save_weights_only=True, verbose=1)
        # tensorboard logs path
        tb_log_dir = log_dir / "logs/scalars/"
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tb_log_dir, histogram_freq=1)

    # Generator config
    train_gen = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                  orientation_type=ORIENTATION, mode='train', val_split=VAL_SPLIT, prediction_target=PREDICTION_TARGET,
                                  add_pos_enc=ADD_POS_ENC, add_depth_map = ADD_DEPTH_MAP)
    val_gen = dp.KittiGenerator(label_dir=LABEL_DIR, image_dir=IMG_DIR, batch_size=BATCH_SIZE,
                                   orientation_type=ORIENTATION, mode='val', val_split=VAL_SPLIT,
                                   all_objs=train_gen.all_objs, prediction_target=PREDICTION_TARGET,
                                   add_pos_enc=ADD_POS_ENC, add_depth_map = ADD_DEPTH_MAP)
    print('Training on {:n} objects. Validating on {:n} objects.'.format(len(train_gen.obj_ids), len(val_gen.obj_ids)))

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

    model.compile(loss=loss_func, loss_weights=loss_weights, optimizer='adam',
                  metrics=OrientationAccuracy(ORIENTATION), run_eagerly=True)


    # early stop callback and accuracy callback
    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss', patience=20)
    if RESUME:
        sub_directories = [path for path in weights_directory.iterdir()]
        if len(sub_directories) == 0:
            raise Exception('No previous training record found. Please enter correct directory or remore --resume option')
        sub_directories.sort()
        latest_training_dir = sub_directories[-1]
        print('-----------------------------------------------')
        print(f'Resume training from directory: {latest_training_dir}')
        weight_files = [str(path) for path in latest_training_dir.iterdir()]
        weight_files.sort()
        latest_weight = weight_files[-1]
        latest_epoch = re.search(r'epoch-(\d\d)-', latest_weight).group(1)
        print(f'Resume training from epoch number: {latest_epoch}')
        init_epoch = int(latest_epoch)
        if not init_epoch: raise Exception("Fail to match epoch number")
        if not os.path.isfile(latest_weight):raise FileNotFoundError(f'stored weights directory "{latest_weight}" is not a valid file')
        model.load_weights(latest_weight)
        # overwrite tensorboard callback
        print(f'current log directory: {logs_directory}')
        tb_log_dir = logs_directory / latest_training_dir.name / "logs" / "scalars"
        if not tb_log_dir.is_dir(): raise FileNotFoundError(f'tensorboard log directory "{tb_log_dir}" is not a valid directory')
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)
        # overwrite call back directory
        cp_callback_file = weights_directory / latest_training_dir / weight_format
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_callback_file, save_weights_only=True, verbose=1)

    train_history = model.fit(x=train_gen, epochs=NUM_EPOCH, verbose=1,
                              validation_data=val_gen, callbacks=[tb_callback, cp_callback], initial_epoch=init_epoch)

    print('Training Finished. Weights and history are saved under directory:', WEIGHT_DIR_ROOT)
    print('Total training time is', timer(start_time, time.time()))