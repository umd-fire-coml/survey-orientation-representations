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
from sklearn.metrics import precision_recall_curve,roc_auc_score
import numpy as np
import random
from statistics import mean
import warnings
parser = argparse.ArgumentParser(description='Draw ROC Curve from predictions file')
# update this to parse str list for comparison chart.
parser.add_argument('pred_paths', metavar = 'pred_paths', type=str,nargs='+',
                    help='Prediction paths to predict and to draw')
parser.add_argument('--pred-dir', dest='pred_dir', type=str, default = './preds',
                    help='The relative paths to directory the predictions are stored in.')      
parser.add_argument('--output-dir', dest='output_dir',type= str, default='./charts',
                   help='The relative path to store the charts')
parser.add_argument('--pred-type',default = 'alpha') # decide if roty or alpha
args = parser.parse_args()


if __name__ == "__main__":
    PREDICTION_PATHS = args.pred_paths
    PREDICTION_DIR = args.pred_dir
    OUTPUT_DIR = args.output_dir
    all_orientations = []
    target_angles = []
    if not os.path.isdir(PREDICTION_DIR):
        raise Exception('kitti_dir is not a directory.')
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    images = {}
    # for each orientation
    for i, path in enumerate(PREDICTION_PATHS):
        ## parse the path
        _, fname = os.path.split(path)
        w_text,ext = os.path.splitext(fname)
        if ext!='.json':
            warnings.warn("Warning: Bad input, ignoring")
            continue
        
        #orientation = ORIENTATIONS[i]
        
        o_path= os.path.join(PREDICTION_DIR, w_text+".json")
        
        
        with open(o_path,"r+") as fp:
            o_preds = json.load(fp,object_hook=json_numpy_obj_hook)
        
        file_args = w_text.strip().split("-")
        
        orientation = file_args[1]
        all_orientations.append(orientation)
        target_angle = file_args[0] #roty or alpha
        target_angles.append(target_angle)
        
        
        if orientation not in ['tricosine','alpha','rot_y','multibin','voting_bin','single_bin']:
            raise Exception("orientation '%s' is not currently supported "%orientation)
        if not os.path.isfile(o_path):
            raise Exception("orientation has no prediction file in :'%s'"%o_path)
            
        for pred in o_preds:
            imgid = pred['img_id']
            if (imgid not in images):
                images[imgid]={}
            if pred['line'] not in images[imgid]:
                images[imgid][pred['line']] ={}
            for ptype in pred['pred']:
                images[imgid][pred['line']][ptype] = pred['pred'][ptype]
        
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
    
    
    for type_obj in ['all','Car','Pedestrian','Cyclist']:
        df = pd.DataFrame(instances)
        print("type: %s"%type_obj)
        filt = df.copy()
        if type_obj == 'all':
            pass
        else:
            
            filt = filt.loc[(df['class'] == type_obj)]
        easy =  df.loc[(df['difficulty'] == 'easy')]
        mod =  df.loc[(df['difficulty'] == 'moderate')]
        hard =  df.loc[(df['difficulty'] == 'hard')]
        for difficulty in ['easy','mod','hard']:
            df = eval(difficulty)
            for i,orientation in enumerate(all_orientations):
                target_angle = target_angles[i]
                deltas = df['target_'+orientation]-df['gt_'+target_angle]
                angle_accuracy = 0.5 * (1.0 + np.cos(deltas))
                num_preds = len(angle_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(angle_accuracy)]
                sorted_alpha = [acc for acc in reversed(sorted(angle_accuracy))]
                pyplot.plot(instance_num, sorted_alpha, marker='.', label=target_angle+'_'+orientation) # predicted angle, model type
                print("%s (target:%s model:%s) mean acc is : %f "%(difficulty,target_angle,orientation,mean(sorted_alpha)))
                
            pyplot.title('Difficulty: '+difficulty)
            # axis labels
            pyplot.xlabel('Instance')
            pyplot.ylabel('Similarity')
            # show the legend
            pyplot.legend()
            # show the plot
            pyplot.savefig(os.path.join(OUTPUT_DIR, type_obj+difficulty+".png"))
            pyplot.clf()
    
    
    

