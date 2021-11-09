import numpy as np
import tensorflow as tf
import math
from orientation_converters import radians_to_angle_normed, radians_to_multibin, radians_to_single_bin, radians_to_tricosine
import metrics
from loss_function import __loss_multibin, __loss_tricosine,  __loss_single_bin_l2,  __loss_alpha_rot_y_angular, __loss_alpha_rot_y_l2
from pprint import pprint
import matplotlib.pyplot as plt
import pathlib
import pandas as pd

# angle_list = np.linspace(-math.pi, math.pi, 10).tolist()
# # voting_bin_list = [angle_to_voting_bin_outputs(angle) for angle in angle_list]
# # test_angle_list = [voting_bin_outputs_to_angle(voting_bin) for voting_bin in voting_bin_list]
# offset_list = [(angle_list[i] - test_angle_list[i]) for i in range(len(test_angle_list))]
#
# print('''
# angle_list: {}
# test_angle_list: {}
# offset_list: {}
# '''.format(
#     angle_list,
#     test_angle_list,
#     offset_list
#     )
# )


if __name__ == "__main__":

    BATCH_SIZE = 30
    # angle_true = np.linspace(-math.pi, math.pi, BATCH_SIZE)
    # angle_pred = angle_true + (np.pi * 2)
    # multibin_true = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in angle_true], dtype=tf.float32)
    # multibin_pred = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in angle_pred], dtype=tf.float32)
    # print('multibin_true:', multibin_true)
    # print('multibin_pred:', multibin_pred)
    # the_metric = metrics.OrientationAccuracy("multibin")

    # the_metric.update_state(multibin_true, multibin_pred)

    # #print("Ground Truth: ", angle_true)
    # print("Accuracy: ", the_metric.result())
    # #print("Offset:", offset)

    y_true = np.full(BATCH_SIZE, 0*np.pi)
    y_pred = np.linspace(-2*math.pi, 2*math.pi, BATCH_SIZE)
    # y_pred = np.linspace(-1,1, BATCH_SIZE)
    print(f'y pred size: {y_pred.shape}')
    # ===== multibin =====
    multibin_true = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in y_true], dtype=tf.float32)
    multibin_pred = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in y_pred], dtype=tf.float32)
    multibin_losses = __loss_multibin(multibin_true, multibin_pred)
    # ===== tricosin =====
    tricosine_true =tf.convert_to_tensor([radians_to_tricosine(angle) for angle in y_true], dtype=tf.float32)
    tricosine_pred =tf.convert_to_tensor([radians_to_tricosine(angle) for angle in y_pred], dtype=tf.float32)
    tricosine_losses = __loss_tricosine(tricosine_true, tricosine_pred)

    # ===== single bin =====
    singlebin_true =tf.convert_to_tensor([radians_to_single_bin(angle) for angle in y_true], dtype=tf.float32)
    singlebin_pred =tf.convert_to_tensor([radians_to_single_bin(angle) for angle in y_pred], dtype=tf.float32)
    singlebin_losses =  __loss_single_bin_l2(singlebin_true, singlebin_pred)
    # ====== angular loss =====
    roty_true =tf.convert_to_tensor([angle for angle in y_true], dtype=tf.float32)
    roty_pred =tf.convert_to_tensor([angle for angle in y_pred], dtype=tf.float32)
    angular_losses =  __loss_alpha_rot_y_angular(roty_true, roty_pred)
    roty_losses = __loss_alpha_rot_y_l2(roty_true, roty_pred)

    for i,(gt, pred) in enumerate(zip(multibin_true, multibin_pred)):
        print(i)
        print(f'GT: {gt}')
        print(f'Pred:{pred}')
        print("---")

    plt.plot(y_pred, multibin_losses)
    plt.savefig("loss_func.png")
    # output_dir = pathlib.Path("loss_function_graph")
    # df = pd.DataFrame(data=np.array([np.asarray(multibin_losses), np.array(tricosine_losses), np.array(singlebin_losses), np.array(angular_losses)]).T, \
    #                     index= np.array(y_pred),
    #                     columns=["Multibin Loss", "Tircosine Loss","SingleBin Loss", "Angular Loss"])
    # df.to_csv(output_dir/"loss_function_data_1pi.csv")
    

