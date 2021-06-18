import argparse, os
import zipfile
import tensorflow as tf

dataset_urls = {
    "training/image_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "training/label_2": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
}

parser = argparse.ArgumentParser(description = 'KITTI Dataset Checker')
parser.add_argument('-f', '--datasetfolder', type = str, help = 'Root Data folder to Check for', default='dataset')
args = parser.parse_args()

for dataset in dataset_urls.keys():
    dataset_path = os.path.abspath(os.path.join(args.datasetfolder, dataset))
    if (os.path.exists(dataset_path) and os.listdir(dataset_path)):
        print(dataset_path, "exists.")
    else:
        print(dataset_path, "does not exist.")
        if (not os.path.exists(args.datasetfolder)):
            os.makedirs(args.datasetfolder)
        url = dataset_urls[dataset]
        print("Downloading dataset to", args.datasetfolder)
        file_path = os.path.abspath(os.path.join(args.datasetfolder, url.rsplit('/', 1)[-1]))
        path_to_downloaded_file = tf.keras.utils.get_file(file_path, url)
        print("Dataset downloaded to ", path_to_downloaded_file)
        with zipfile.ZipFile(path_to_downloaded_file, 'r') as zip_ref:
            zip_ref.extractall(args.datasetfolder)
            print("Dataset extracted.")
        if os.path.exists(path_to_downloaded_file):
            os.remove(path_to_downloaded_file)
