'''Run this file to get the accuracy distribution in a png file, outputs to default directory ./charts/'''

# init code
from testing import PREDICTION_TARGET


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
        
parser = argparse.ArgumentParser(description='Draw ROC Curve from predictions file')
# update this to parse str list for comparison chart.
parser.add_argument(dest='orientations', type=[],
                    help='Orientation Type of the model. Options are tricosine, alpha, rot_y, multibin, voting_bin, single_bin')
parser.add_argument(dest='pred_files', type=[],
                    help='The relative paths to the predictions json file obtained from testing.py.')      
parser.add_argument('--output-dir', dest='output_dir',tyepe= str,default='charts',
                   help='The relative path to store the charts')
args = parser.parse_args()


if __name__ == "__main__":
    ORIENTATIONS = args.orientations
    PREDICTION_FILEPATHS = args.pred_files
    OUTPUT_DIR = args.output_dir

    # for each orientation
    for orientation in ORIENTATIONS:
        # overlay accuracy distribution
        # label each graph with mean accuracy

