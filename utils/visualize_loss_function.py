import numpy as np
import tensorflow as tf
import sys, math
import pathlib
# print(f'resolved path: {pathlib.Path().resolve()}')
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from model.orientation_converters import *
import model.metrics
import model.loss_function as loss_function
from pprint import pprint
import matplotlib.pyplot as plt

import pandas as pd

def generate_loss_csv(batch_size, ground_truth, visualize = False):
    y_true = np.full(batch_size, ground_truth)
    y_pred = np.linspace(-2*math.pi, 2*math.pi, batch_size)
    output_batch = np.zeros((7, batch_size))
    # ===== multibin =====
    multibin_scaling_factor = 0.5
    multibin_true = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in y_true], dtype=tf.float32)
    multibin_pred = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in y_pred], dtype=tf.float32)
    # multibin_true = tf.convert_to_tensor([np.concatenate(radians_to_multi_affinity_bin(angle), axis=-1) for angle in y_true], dtype=tf.float32)
    # multibin_pred = tf.convert_to_tensor([np.concatenate(radians_to_multi_affinity_bin(angle), axis=-1) for angle in y_pred], dtype=tf.float32)
    multibin_losses = multibin_scaling_factor*loss_function.__loss_multibin(multibin_true, multibin_pred)
    output_batch[0,:] = multibin_losses
    # ===== tricosin =====
    tricosine_true =tf.convert_to_tensor([radians_to_tricosine(angle) for angle in y_true], dtype=tf.float32)
    tricosine_pred =tf.convert_to_tensor([radians_to_tricosine(angle) for angle in y_pred], dtype=tf.float32)
    tricosine_losses = loss_function.__loss_tricosine(tricosine_true, tricosine_pred)
    output_batch[1,:] = tricosine_losses
    # ===== single bin =====
    singlebin_true =tf.convert_to_tensor([radians_to_single_bin(angle) for angle in y_true], dtype=tf.float32)
    singlebin_pred =tf.convert_to_tensor([radians_to_single_bin(angle) for angle in y_pred], dtype=tf.float32)
    singlebin_losses =  loss_function.__loss_single_bin_l2(singlebin_true, singlebin_pred)
    output_batch[2,:] = singlebin_losses
    # ===== voting bin =====
    votingbin_true = tf.convert_to_tensor([np.concatenate(radians_to_voting_bin(angle), -1) for angle in y_true], dtype=tf.float32)
    votingbin_pred = tf.convert_to_tensor([np.concatenate(radians_to_voting_bin(angle), -1) for angle in y_true], dtype=tf.float32)
    votingbin_losses =  loss_function.__loss_voting_bin(votingbin_true, votingbin_pred)
    output_batch[3,:] = votingbin_losses
    # ====== angular and roty loss ======
    angular_loss_scaling_factor = 50
    roty_true =tf.convert_to_tensor([angle for angle in y_true], dtype=tf.float32)
    roty_pred =tf.convert_to_tensor([angle for angle in y_pred], dtype=tf.float32)
    angular_losses = angular_loss_scaling_factor * loss_function.__loss_alpha_rot_y_angular(roty_true, roty_pred)
    output_batch[4,:] = angular_losses
    roty_losses = loss_function.__loss_alpha_rot_y_l2(roty_true, roty_pred)
    output_batch[5,:] = roty_losses
    # ====== l2 loss ======
    l2_loss = loss_function.l2_loss(y_true, y_pred)
    output_batch[6,:] = l2_loss
    if visualize:
        plt.figure()
        plt.plot(y_pred, multibin_losses)
        plt.savefig(f"gt_{str(np.round(ground_truth,3))}_loss_func.png")
    
    return output_batch.T, y_pred

if __name__ == "__main__":

    BATCH_SIZE = 900
    df_list = []
    for gt in [0, 0.5*math.pi, math.pi]:    
        loss_array, y_pred = generate_loss_csv(BATCH_SIZE, gt, True)
        output_dir = pathlib.Path("../loss_function_graph")
        df = pd.DataFrame(data=loss_array, \
                            index= np.array(y_pred),
                            columns=["Multibin Loss", "Tircosine Loss","SingleBin Loss","VotingBin Loss", "Angular Loss","RotY Loss", "L2 Loss",])
        df_list.append(df)
    all_df = pd.concat(df_list, 1)
    all_df.to_csv(output_dir/"loss_function.csv")
    

