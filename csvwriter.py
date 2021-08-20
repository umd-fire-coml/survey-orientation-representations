import pandas as pd
from serialize import NumpyEncoder,json_numpy_obj_hook
import json
import orientation_converters as conv
from tqdm import tqdm
import math
import os
import warnings



warnings.warn("Warning: please ensure that there is only at most one angle of each type")
'''
ensure the predictions are in preds
'''
sygyzy={}
for ppath in os.listdir('preds/'):
    w_text,ext = os.path.splitext(ppath)
    if ext=='.json':
        with open("preds/"+ppath,"r+") as fp:
            loaded_json = json.load(fp,object_hook=json_numpy_obj_hook)
        loaded_json = loaded_json[1:]
        for p in loaded_json:
            imgid = p['img_id']
            if (imgid not in sygyzy):
                sygyzy[imgid]={}
            if (p['line'] not in sygyzy[imgid]):
                sygyzy[imgid][p['line']] = {}
            for key in p['pred']:
                sygyzy[imgid][p['line']][key] = p['pred'][key]
        
base = []
for c,imgid in enumerate(sygyzy):
    for instance in sygyzy[imgid]:
        work = sygyzy[imgid][instance]
        work['img_id'] = imgid
        work['line'] = instance
        tokens = instance.strip().split(' ')
        work['gt_alpha'] = float(tokens[3])
        work['gt_rot_y'] = float(tokens[14])
        work['class']=tokens[0]
        height = float(tokens[7])-float(tokens[5])
        occlusion = int(tokens[2])
        truncation = float(tokens[1])
        if (height>40 and occlusion<=0 and truncation<.15):
            work['difficulty'] = 'easy'
        elif (height>25 and occlusion<=1 and truncation<.3):
            work['difficulty'] = 'moderate'
        elif (height>25 and occlusion<=2 and truncation<.5):
            work['difficulty'] = 'hard'
        base.append(work)
df = pd.DataFrame(base)
df.to_csv("preds/out.csv")
print("Done!")