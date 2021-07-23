'''Run this file to get the accuracy distribution in a png file, outputs to default directory ./charts/'''
'''This assumes that all the preds are in ./pred_dir/'''
import pandas as pd
from serialize import NumpyEncoder,json_numpy_obj_hook
import json
import orientation_converters as conv
from tqdm import tqdm
import math
from matplotlib import pyplot
import argparse
import os


parser = argparse.ArgumentParser(description='Draw ROC Curve from predictions file')
# update this to parse str list for comparison chart.
parser.add_argument('orientations', metavar = 'orientation', type=str,nargs='+',
                    help='Orientation Type of the model. Options are tricosine, alpha, rot_y, multibin, voting_bin, single_bin')
parser.add_argument('--pred-dir', dest='pred_dir', type=str, default = './preds',
                    help='The relative paths to directory the predictions are stored in.')      
parser.add_argument('--output-dir', dest='output_dir',type= str, default='./charts',
                   help='The relative path to store the charts')
args = parser.parse_args()


if __name__ == "__main__":
    ORIENTATIONS = args.orientations
    PREDICTION_DIR = args.pred_dir
    OUTPUT_DIR = args.output_dir
    if not os.path.isdir(PREDICTION_DIR):
        raise Exception('kitti_dir is not a directory.')
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    images = {}
    # for each orientation
    for orientation in ORIENTATIONS:
        o_path= os.path.join(PREDICTION_DIR, orientation+".json")
        if orientation not in ['tricosine','alpha','rot_y','multibin','voting_bin','single_bin']:
            raise Exception("orientation '%s' is not currently supported "%orientation)
        if not os.path.isfile(o_path):
            raise Exception("orientation has no prediction file in :'%s'"%o_path)
        
        with open(o_path,"r+") as fp:
            o_preds = json.load(fp,object_hook=json_numpy_obj_hook)
        
        if orientation=='alpha':
            for p in o_preds:
            # get the first value, i.e. the alpha prediction
                pred_alpha = conv.angle_normed_to_radians(p['pred'][0])
                imgid = p['img_id']
                if (imgid not in images):
                    images[imgid]={}
                if p['line'] not in images[imgid]:
                    images[imgid][p['line']] ={}
                images[imgid][p['line']]['norm_alpha'] =p['pred'][0]
                images[imgid][p['line']]['pred_alpha'] = pred_alpha
        elif orientation == 'rot_y':
            for p in o_preds:
                # get the first value, i.e. the rot_y prediction
                pred_roty = conv.angle_normed_to_radians(p['pred'][0])
                imgid = p['img_id']
                if (imgid not in images):
                    raise
                if p['line'] not in images[imgid]:
                    images[imgid][p['line']] ={}
                images[imgid][p['line']]['norm_roty'] = p['pred'][0]
                images[imgid][p['line']]['pred_roty'] = pred_roty
                tokens = p['line'].strip().split(' ')
                images[imgid][p['line']]['conv_roty'] = conv.rot_y_to_alpha(pred_roty,float(tokens[11]),float(tokens[13]))
        elif orientation == 'single_bin':
            for p in o_preds:
                conv_single = conv.single_bin_to_radians(p['pred'])
                imgid = p['img_id']
                if (imgid not in images):
                    raise
                if p['line'] not in images[imgid]:
                    images[imgid][p['line']] ={}
                images[imgid][p['line']]['single_pred'] = p['pred']
                images[imgid][p['line']]['conv_single'] = conv_single
        elif orientation == 'voting_bin':
            for p in o_preds:
                conv_voting = conv.voting_bin_to_radians(p['pred'])
                imgid = p['img_id']
                if (conv_voting>math.pi):
                    conv_voting-=math.tau
                if (imgid not in images):
                    raise
                if p['line'] not in images[imgid]:
                    images[imgid][p['line']] ={}
                images[imgid][p['line']]['voting_pred'] = p['pred']
                images[imgid][p['line']]['conv_voting'] = conv_voting
        elif orientation ='tricosine':
            for p in o_preds:
                conv_tri = conv.tricosine_to_radians(p['pred'])
                imgid = p['img_id']
                if (imgid not in images): # create a dict for each img_id
                    raise
                if p['line'] not in images[imgid]:
                    images[imgid][p['line']] ={}
                images[imgid][p['line']]['tricosine_pred'] = p['pred']
                images[imgid][p['line']]['conv_tricosine'] = conv_tri

        # overlay accuracy distribution
        # label each graph with mean accuracy
    instances = []
    for c,imgid in enumerate(images):
        for instance in images[imgid]:
            instance_dict = images[imgid][instance]
            # add more keys to instance_dict
            instance_dict['imgid'] = imgid
            instance_dict['line'] = instance
            tokens = instance.strip().split(' ')
            instance_dict['gt_alpha'] = float(tokens[3])
            instance_dict['gt_rot_y'] = float(tokens[14])
            instance_dict['class']=tokens[0]
            height = float(tokens[7])-float(tokens[5])
            occlusion = int(tokens[2])
            truncation = float(tokens[1])
            if (height>40 and occlusion<=0 and truncation<.15):
                instance_dict['difficulty'] = 'easy'
            elif (height>25 and occlusion<=1 and truncation<.3):
                instance_dict['difficulty'] = 'moderate'
            elif (height>25 and occlusion<=2 and truncation<.5):
                instance_dict['difficulty'] = 'hard'
            instances.append(instance_dict)
    df = pd.DataFrame(instances)
    
    
    

