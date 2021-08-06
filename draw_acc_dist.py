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
            if 'alpha' in ORIENTATIONS:
                deltas = df['pred_alpha']-df['gt_alpha']
                alpha_accuracy = 0.5 * (1.0 + np.cos(deltas))
                num_preds = len(alpha_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(alpha_accuracy)]
                sorted_alpha = [acc for acc in reversed(sorted(alpha_accuracy))]
                pyplot.plot(instance_num, sorted_alpha, marker='.', label='Alpha')
                print("%s alpha overall is : %f "%(difficulty,mean(sorted_alpha)))
            if 'rot_y' in ORIENTATIONS:
                deltas = df['conv_roty']-df['gt_alpha']
                roty_accuracy = (1+np.cos(deltas))/2
                num_preds = len(roty_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(roty_accuracy)]
                sorted_roty = [i for i in reversed(sorted(roty_accuracy ))]
                pyplot.plot(instance_num, sorted_roty, marker='.', label='Rot_y')
                print("%s rot_y overall is : %f"%(difficulty,mean(sorted_roty)))
            if 'single_bin' in ORIENTATIONS:
                deltas = df['conv_single']-df['gt_alpha']
                single_accuracy = (1+np.cos(deltas))/2
                num_preds = len(single_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(single_accuracy)]
                sorted_single = [acc for acc in reversed(sorted(single_accuracy))]
                pyplot.plot(instance_num, sorted_single, marker='.', label='Single')
                print("%s single bin overall is : %f"%(difficulty,mean(sorted_single)))
            if 'voting_bin' in ORIENTATIONS:
                deltas = df['conv_voting']-df['gt_alpha']
                voting_accuracy = (1+np.cos(deltas))/2
                num_preds = len(voting_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(voting_accuracy)]
                sorted_voting = [i for i in reversed(sorted(voting_accuracy))]
                pyplot.plot(instance_num, sorted_voting, marker='.', label='Voting')
                print("%s sorted bin overall is : %f"%(difficulty, mean(sorted_voting)))
            if 'tricosine' in ORIENTATIONS:
                deltas = df['conv_tricosine']-df['gt_alpha']
                tri_accuracy = (1+np.cos(deltas))/2
                num_preds = len(tri_accuracy)
                instance_num = [idx/num_preds for idx, acc in enumerate(tri_accuracy)]
                sorted_tri = [i for i in reversed(sorted(tri_accuracy))]
                pyplot.plot(instance_num, sorted_tri, marker='.', label='Tricosine')
                print("%s tricosine overall is : %f"%(difficulty,mean(sorted_tri)))
            pyplot.title('Difficulty: '+difficulty)
            # axis labels
            pyplot.xlabel('Instance')
            pyplot.ylabel('Similarity')
            # show the legend
            pyplot.legend()
            # show the plot
            pyplot.savefig(os.path.join(OUTPUT_DIR, type_obj+difficulty+".png"))
            pyplot.clf()
    
    
    

