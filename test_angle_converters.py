import numpy as np
import tensorflow as tf
import math
from orientation_converters import radians_to_multibin, radians_to_tricosine, radians_to_voting_bin, voting_bin_to_radians, multibin_orientation_confidence_to_radians, MULTIBIN_SHAPE, radians_to_angle_normed
import metrics
from loss_function import __loss_multibin
from pprint import pprint
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

    BATCH_SIZE = 10
    angle_true = np.linspace(-math.pi, math.pi, BATCH_SIZE)
    angle_pred = angle_true + (np.pi * 2)
    multibin_true = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in angle_true], dtype=tf.float32)
    multibin_pred = tf.convert_to_tensor([np.concatenate(radians_to_multibin(angle), axis=-1) for angle in angle_pred], dtype=tf.float32)
    print('multibin_true:', multibin_true)
    print('multibin_pred:', multibin_pred)
    the_metric = metrics.OrientationAccuracy("multibin")

    the_metric.update_state(multibin_true, multibin_pred)

    #print("Ground Truth: ", angle_true)
    print("Accuracy: ", the_metric.result())
    #print("Offset:", offset)
