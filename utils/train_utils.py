import tensorflow as tf
import os, pathlib
import regex as re

def setup_gpu():
    # set up tensorflow GPU
    tf.config.list_physical_devices('GPU')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            config = tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                # device_count = {'GPU': 1}
            )
            config.gpu_options.allow_growth = True
            session = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(session)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# this static method take a python parser as input
def setup_cmd_arg(parser):
    parser.add_argument(
    '--predict',
    dest='predict',
    type=str,
    default="rot-y",
    help='The target angle to be predicted. Options are rot-y, alpha',
    )
    parser.add_argument(
        '--converter',
        dest='orientation',
        type=str,
        help=(
            'Orientation conversion type of the model. '
            'Options are alpha, rot-y, tricosine, multibin, voting-bin, single-bin'
        ),
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        default=8,
        help='Define the batch size for training. Default value is 8',
    )
    parser.add_argument(
        '--epoch',
        dest='num_epoch',
        type=int,
        default=100,
        help='Number of epoch used for training. Default value is 100',
    )
    parser.add_argument(
        '--kitti_dir',
        dest='kitti_dir',
        type=str,
        default='dataset',
        help=(
            'path to kitti dataset directory. Its subdirectory should have training/ and testing/.'
            ' Default path is dataset/'
        ),
    )
    parser.add_argument(
        '--training_record',
        dest='training_record',
        type=str,
        default='training_record',
        help=(
            'root directory of all training record, parent of weights and logs directory. '
            'Default path is training_record'
        ),
    )
    parser.add_argument(
        '--log_dir',
        dest='log_dir',
        type=str,
        help='path to tensorboard logs directory. Default path is training_record/logs',
    )
    parser.add_argument(
        '--weight_dir',
        dest='weight_dir',
        type=str,
        help='Relative path to save weights. Default path is training_record/weights',
    )
    parser.add_argument(
        '--val_split',
        dest='val_split',
        type=float,
        default=0.2,
        help='Fraction of the dataset used for validation. Default val_split is 0.2',
    )
    parser.add_argument(
        '--resume',
        dest='resume',
        type=bool,
        default=False,
        help='Resume from previous training under training_record directory',
    )
    parser.add_argument(
        '--add_pos_enc',
        dest='add_pos_enc',
        type=bool,
        default=False,
        help='Add positional encoding to input',
    )
    parser.add_argument("--use_angular_loss", dest='use_angular_loss', type=bool, default=False)
    parser.add_argument(
        "--add_depth_map",
        dest="add_depth_map",
        type=bool,
        default=False,
        help="If add_depth_map is true, add the path to directory containing depth map.",
    )
    args = parser.parse_args()
    return args

def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

def check_args(args, depth_path_dir, label_dir, img_dir):

    if not os.path.isdir(args.kitti_dir):
        raise Exception('kitti_dir is not a directory.')
    if args.orientation not in [
        'tricosine',
        'alpha',
        'rot-y',
        'multibin',
        'voting-bin',
        'single-bin',
        'exp-A'
    ]:
        raise Exception('Invalid Orientation Type.')
    if not 0.0 <= args.val_split <= 1.0:
        raise Exception('Invalid val_split range between [0.0, 1.0]')
    if args.add_depth_map and (not os.path.isdir(depth_path_dir)):
        raise Exception(
            "Unable to find depth maps. Please put depth map under"
            " /kitti_dataset/training/predic_depth"
        )

def find_latest_epoch_and_weights(weights_directory, verbose = True):
    sub_directories = [path for path in weights_directory.iterdir()]
    if len(sub_directories) == 0:
        raise Exception(
            'No previous training record found. Please enter correct directory or remore'
            ' --resume option'
        )
    sub_directories.sort()
    latest_training_dir = sub_directories[-1]
    weight_files = [str(path) for path in latest_training_dir.iterdir()]
    weight_files.sort()
    latest_weight = weight_files[-1]
    latest_epoch = re.search(r'epoch-(\d\d)-', latest_weight).group(1)
    if verbose:
        print('-----------------------------------------------')
        print(f'Resume training from directory: {latest_training_dir}')
        print(f'Resume training from epoch number: {latest_epoch}')
    return latest_training_dir, latest_epoch, latest_weight