import pandas as pd
from serialize import NumpyEncoder,json_numpy_obj_hook
import json
import orientation_converters as conv
from tqdm import tqdm
import math
'''
yes its your job to make sure these exist
'''

with open("preds/alpha_normed.json","r+") as fp:
    alpha_preds = json.load(fp,object_hook=json_numpy_obj_hook)
with open("preds/rot_y_normed.json","r+") as fp:
    roty_preds = json.load(fp,object_hook=json_numpy_obj_hook)
with open("preds/single_bin_normed.json","r+") as fp:
    single_preds = json.load(fp,object_hook=json_numpy_obj_hook)
with open("preds/voting_bin_normed.json","r+") as fp:
    voting_preds = json.load(fp,object_hook=json_numpy_obj_hook)
with open("preds/tricosine_normed.json","r+") as fp:
    tricosine_preds = json.load(fp,object_hook=json_numpy_obj_hook)
sygyzy={}
for p in alpha_preds:
    pred_alpha = conv.angle_normed_to_radians(p['pred'][0])
    imgid = p['img']
    if (imgid not in sygyzy):
        sygyzy[imgid]={}
    sygyzy[imgid][p['line']] ={'norm_alpha':p['pred'][0],'pred_alpha':pred_alpha}
for p in roty_preds:
    pred_roty = conv.angle_normed_to_radians(p['pred'][0])
    imgid = p['img']
    if (imgid not in sygyzy):
        raise
    sygyzy[imgid][p['line']]['norm_roty'] = p['pred'][0]
    sygyzy[imgid][p['line']]['pred_roty'] = pred_roty
    tokens = p['line'].strip().split(' ')
    sygyzy[imgid][p['line']]['conv_roty'] = conv.rot_y_to_alpha(pred_roty,float(tokens[11]),float(tokens[13]))
    
for p in single_preds:
    conv_single = conv.single_bin_to_radians(p['pred'])
    imgid = p['img']
    if (imgid not in sygyzy):
        raise
    sygyzy[imgid][p['line']]['single_pred'] = p['pred']
    sygyzy[imgid][p['line']]['conv_single'] = conv_single
for p in voting_preds:
    conv_voting = conv.voting_bin_to_radians(p['pred'])
    imgid = p['img']
    if (conv_voting>math.pi):
        conv_voting-=math.tau
    if (imgid not in sygyzy):
        raise
    sygyzy[imgid][p['line']]['voting_pred'] = p['pred']
    sygyzy[imgid][p['line']]['conv_voting'] = conv_voting
for c,p in enumerate(tricosine_preds):
    conv_tri = conv.tricosine_to_radians(p['pred'])
    imgid = p['img']
    if (imgid not in sygyzy):
        raise
    sygyzy[imgid][p['line']]['tricosine_pred'] = p['pred']
    sygyzy[imgid][p['line']]['conv_tricosine'] = conv_tri
base = []
for c,imgid in enumerate(sygyzy):
    for instance in sygyzy[imgid]:
        work = sygyzy[imgid][instance]
        work['imgid'] = imgid
        work['line'] = instance
        tokens = instance.strip().split(' ')
        work['gt_alpha'] = tokens[3]
        work['gt_rot_y'] = tokens[14]
        base.append(work)
df = pd.DataFrame(base)
df.to_csv("preds/out.csv")
print("Done!")