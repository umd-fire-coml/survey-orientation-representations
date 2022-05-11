from math import tau
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import math

'''All methods here assume a single element of input, not a numpy array of inputs'''

# global constants
TAU = np.pi * 2.0

# global helper methods
def is_angle_between(angle, start, end):
    wedge = (end - start) % TAU
    offset_from_start = (angle - start) % TAU
    offset_from_end = (end - angle) % TAU
    return offset_from_start < wedge and offset_from_end < wedge


def get_mean_angle(batch_radians):
    # calculuate the mean of angles, output between [-pi, pi]
    arr_angles = np.asarray(batch_radians)
    sum_sin_predicted_angles = np.sum(np.sin(arr_angles))
    sum_cos_predicted_angles = np.sum(np.cos(arr_angles))
    return np.arctan2(sum_sin_predicted_angles, sum_cos_predicted_angles)


# trisector affinity constants
SECTORS = int(3)
SECTOR_WIDTH = TAU / SECTORS
HALF_SECTOR_WIDTH = SECTOR_WIDTH / 2
SHAPE_TRICOSINE = (SECTORS,)


def radians_to_tricosine(angle_rad):
    """Return a numpy array of trisector affinity values from an angle (such as alpha or rot_y) in radians

    Key Properties:
    - output represent the affinity value (cos distance) to the middle of 3 sectors
    - affinity increases if an angle moves towards the middle of each sector
    - affinity decreases if an angle moves away from the middle of each sector
    - if the angle is at the middle of a sector, output 1
    - if the angle is at the exact opposite of the sector center, output -1
    - affinity for each sector is within the range [0.0-1.0]
    """

    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU

    # output array
    trisector_affinity = np.empty(shape=SHAPE_TRICOSINE)

    # calculate the bounding sector affinity
    # get the bounding sector number in which the angle is within the bounds of sector's start and end
    bounding_sector_num = int(new_angle_rad // SECTOR_WIDTH)
    # get the bounding sector's start position
    bounding_sector_start = SECTOR_WIDTH * bounding_sector_num
    # get the bounding sector's mid position
    bounding_sector_mid = bounding_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the the middle of the bounding sector
    offset_from_bounding_sector_mid = new_angle_rad - bounding_sector_mid
    # get the sector affinity based on the offset
    bounding_sector_affinity = np.cos(offset_from_bounding_sector_mid)
    # insert to output array
    trisector_affinity[bounding_sector_num] = bounding_sector_affinity

    # calculate the left sector affinity
    # get the left sector num, simply minus 1 then wrap
    left_sector_num = (bounding_sector_num - 1) % SECTORS  # if -1 then we get 2
    # get the left sector's start position
    left_sector_start = SECTOR_WIDTH * left_sector_num
    # get the left sector's mid position
    left_sector_mid = left_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the start of the bounding sector
    offset_from_bounding_sector_start = new_angle_rad - bounding_sector_start
    # get how much is the angle offset from the the middle of the left sector
    offset_from_left_sector_mid = HALF_SECTOR_WIDTH + offset_from_bounding_sector_start
    # get the sector affinity based on the offset
    left_sector_affinity = np.cos(offset_from_left_sector_mid)
    # insert to output array
    trisector_affinity[left_sector_num] = left_sector_affinity

    # calculate the right sector affinity
    # get the right sector num, simply plus 1 then wrap
    right_sector_num = (bounding_sector_num + 1) % SECTORS  # if 3 we get 0
    # get the right sector's start position
    right_sector_start = SECTOR_WIDTH * right_sector_num
    # get the right sector's mid position
    right_sector_mid = right_sector_start + HALF_SECTOR_WIDTH
    # get how much is the angle offset from the end of the bounding sector
    offset_from_bounding_sector_end = SECTOR_WIDTH - offset_from_bounding_sector_start
    # get how much is the angle offset from the the middle of the right sector
    offset_from_right_sector_mid = HALF_SECTOR_WIDTH + offset_from_bounding_sector_end
    # get the sector affinity based on the offset
    right_sector_affinity = np.cos(offset_from_right_sector_mid)
    # insert to output array
    trisector_affinity[right_sector_num] = right_sector_affinity

    return trisector_affinity


def tricosine_to_radians(trisector_affinity, allow_negative_pi=True):
    """Return an angle in radians from trisector affinity, allow_negative_pi sets the output range [-pi to +pi]"""

    # clip values between -1 and 1 for acos.
    trisector_affinity = np.clip(trisector_affinity, -1.0, 1.0)

    # calculate the possible angles based on bounding sector offset
    # get the bounding sector number
    bounding_sector_num = np.argmax(trisector_affinity)
    # get the bounding sector's start position
    bounding_sector_start = SECTOR_WIDTH * bounding_sector_num
    # get the bounding sector's mid position
    bounding_sector_mid = bounding_sector_start + HALF_SECTOR_WIDTH
    # get bounding sector affinity
    bounding_sector_affinity = trisector_affinity[bounding_sector_num]
    # get how much is the angle offset from the the middle of the bounding sector
    offset_from_bounding_sector_mid = np.arccos(bounding_sector_affinity)
    # get the two possible angles based on offset_from_bounding_sector_mid
    l_angle_from_bounding_sector_offset = bounding_sector_mid - offset_from_bounding_sector_mid
    r_angle_from_bounding_sector_offset = bounding_sector_mid + offset_from_bounding_sector_mid

    # calculate the possible angle based on left sector offset
    # get the left sector num, simply minus 1 then wrap
    left_sector_num = (bounding_sector_num - 1) % SECTORS  # if -1 then we get 2
    # get the left sector's start position
    left_sector_start = SECTOR_WIDTH * left_sector_num
    # get the left sector's mid position
    left_sector_mid = left_sector_start + HALF_SECTOR_WIDTH
    # get left sector affinity
    left_sector_affinity = trisector_affinity[left_sector_num]
    # get how much is the angle offset from the the middle of the left sector
    offset_from_left_sector_mid = np.arccos(left_sector_affinity)
    # get the predicted angle based on offset_from_left_sector_mid then wrap, if tau+1 then 1
    # left sector will always offset towards the right
    predicted_angle_from_left_sector_offset = (
        left_sector_mid + offset_from_left_sector_mid
    ) % TAU

    # calculate the possible angle based on left sector offset
    # get the right sector num, simply plus 1 then wrap
    right_sector_num = (bounding_sector_num + 1) % SECTORS  # if 3 we get 0
    # get the right sector's start position
    right_sector_start = SECTOR_WIDTH * right_sector_num
    # get the right sector's mid position
    right_sector_mid = right_sector_start + HALF_SECTOR_WIDTH
    # get right sector affinity
    right_sector_affinity = trisector_affinity[right_sector_num]
    # get how much is the angle offset from the the middle of the right sector
    offset_from_right_sector_mid = np.arccos(right_sector_affinity)
    # get the predicted angle based on offset_from_right_sector_mid then wrap, if -1 then tau-1
    # left sector will always offset towards the right
    predicted_angle_from_right_sector_offset = (
        right_sector_mid - offset_from_right_sector_mid
    ) % TAU

    # get the predicted angle from bounding sector (based on left right offset signals)
    if offset_from_left_sector_mid < offset_from_right_sector_mid:
        predicted_angle_from_bounding_sector_offset = l_angle_from_bounding_sector_offset
    else:
        predicted_angle_from_bounding_sector_offset = r_angle_from_bounding_sector_offset

    # calculuate the mean of predicted angles
    mean_angle = get_mean_angle(
        [
            predicted_angle_from_left_sector_offset,
            predicted_angle_from_right_sector_offset,
            predicted_angle_from_bounding_sector_offset,
        ]
    )

    if allow_negative_pi:
        return mean_angle
    else:
        return mean_angle % TAU


# multibin constants
NUM_BIN = int(4)
OVERLAP = 0.1
BIN_SIZE = TAU / NUM_BIN  # angle size of each bin, i.e. 180 deg
BIN_EXT = (OVERLAP / 2) * BIN_SIZE  # extension at start and end
ORIENTATION_SHAPE = (NUM_BIN, 2)
CONFIDENCE_SHAPE = (NUM_BIN, 1)
SHAPE_MULTIBIN = (NUM_BIN, 3)


def radians_to_multibin(angle_rad):
    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU
    # set all values as zeros for each orientation (2x2 values) and conf  (2 values, each value represents the sector)
    orientation = np.zeros(ORIENTATION_SHAPE)
    confidence = np.zeros(CONFIDENCE_SHAPE)

    # get the sector bounding points
    for bin_id in range(NUM_BIN):
        # BIN START
        BIN_START = bin_id * BIN_SIZE - BIN_EXT
        BIN_END = (bin_id + 1) * BIN_SIZE + BIN_EXT
        if is_angle_between(new_angle_rad, BIN_START, BIN_END):
            angle_bin_start_offset = new_angle_rad - BIN_START
            # calculate bin affinity with cos and sin
            orientation[bin_id] = np.asarray(
                [np.cos(angle_bin_start_offset), np.sin(angle_bin_start_offset)]
            )
            # set bin confidence to 1
            confidence[bin_id] = np.asarray([1.0])

    # if in both sectors, then each confidence is 1/2, this makes sure sum of confidence adds up to 1
    # handle 0 division:
    # confidence_sum = np.sum(confidence) if (np.sum(confidence) != 0) else 1e-10
    confidence_sum = np.sum(confidence)
    confidence = confidence / confidence_sum

    return orientation, confidence


def multibin_orientation_confidence_to_radians(
    orientation, confidence, allow_negative_pi=True
):
    # clip values between -1 and 1 for acos.
    orientation = tf.clip_by_value(orientation, -1.0, 1.0)
    predicted_angles = (
        []
    )  # tensorflow object doesn't support assignment. We have to stack them together
    predicted_confs = []

    # get predictions and conf from each bin
    for bin_id in range(NUM_BIN):
        # get the angle
        cos = orientation[bin_id, 0]
        sin = orientation[bin_id, 1]
        angle_bin_start_offset = tf.math.atan2(sin, cos)
        BIN_START = bin_id * BIN_SIZE - BIN_EXT
        predicted_angle = angle_bin_start_offset + BIN_START
        # force angles in [0, 2pi] range for average calculation
        # predicted_angles[bin_id] = tf.math.mod(predicted_angle, TAU)
        predicted_angles.append(tf.math.mod(predicted_angle, TAU))
        # get the confidence
        bin_conf = confidence[bin_id, 0]
        # predicted_confs[bin_id] = bin_conf
        predicted_confs.append(bin_conf.numpy().item())
    # get the prediction from bin with highest confidence
    pred_angle = predicted_angles[tf.math.argmax(predicted_confs).numpy().item()]

    # convert pred_angle to kitti range
    if allow_negative_pi:
        # print(f'type of pred_angle: {type(pred_angle)}')
        pred_angle = tf.map_fn(lambda x: x if x < math.pi else x - TAU, pred_angle)
        # return pred_angle if (pred_angle < math.pi) else pred_angle - TAU
        # pred_angle = pred_angle if pred_angle < math.pi else pred_angle - math.pi
        return pred_angle

    else:
        return pred_angle


def multibin_to_radians(multibin):
    # print(f'__debug__ multibin shape: {multibin.shape}')
    return multibin_orientation_confidence_to_radians(multibin[..., :2], multibin[..., 2:])


def batch_multibin_to_batch_radians(batch_multibin):
    # print(f'__debug__ batch_multibin shape: {batch_multibin.shape}')
    return tf.map_fn(multibin_to_radians, batch_multibin)


# alpha and rot_y constants
ALPHA_ROT_Y_NORM_FACTOR = tf.constant(np.pi, dtype=tf.float64)
SHAPE_ALPHA_ROT_Y = (1,)


def radians_to_angle_normed(angle_rad):
    '''normalize angle_rad [-pi, pi] to [-1,1]'''
    return angle_rad / ALPHA_ROT_Y_NORM_FACTOR


def angle_normed_to_radians(angle_normed):
    return ALPHA_ROT_Y_NORM_FACTOR * tf.cast(angle_normed, dtype=tf.float64)


def alpha_to_rot_y(alpha, loc_x, loc_z):
    return alpha + np.arctan(loc_x / loc_z)


# single bin and voting bin constants
SHAPE_SINGLE_BIN = (2,)
NUM_OF_VOTING_BINS = 4  # minimum is 3
VOTING_BIN_WIDTH = TAU / NUM_OF_VOTING_BINS
SHAPE_VOTING_BIN = (NUM_OF_VOTING_BINS, 2)
VOTING_BIN_THRESHOLD = 60.0 / 360.0 * TAU


def radians_to_single_bin(angle_rad):
    return np.asarray([np.cos(angle_rad), np.sin(angle_rad)])


def single_bin_to_radians(orientation, allow_negative_pi=True):
    angle = np.arctan2(orientation[1], orientation[0])
    if allow_negative_pi:
        return angle
    else:
        return angle % TAU


def radians_to_voting_bin(angle_rad):
    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU

    # placeholder for all output values
    orientation = np.zeros(shape=SHAPE_VOTING_BIN)

    for bin_id in range(NUM_OF_VOTING_BINS):
        # BIN START
        BIN_START = bin_id * BIN_SIZE
        BIN_CENTER = BIN_START + (VOTING_BIN_WIDTH / 2)
        angle_bin_center_offset = new_angle_rad - BIN_CENTER
        # calculate bin affinity with cos and sin
        orientation[bin_id] = radians_to_single_bin(angle_bin_center_offset)

    return orientation


def voting_bin_to_radians(orientation):
    # clip values between -1 and 1 for acos.
    orientation = np.clip(orientation, -1.0, 1.0)

    # placeholder for the list of all predictions
    predicted_angles = np.empty(shape=(NUM_OF_VOTING_BINS,))

    # get the weighted average of all predictions
    for bin_id in range(NUM_OF_VOTING_BINS):
        # get the angle
        angle_bin_center_offset = single_bin_to_radians(orientation[bin_id])
        BIN_START = bin_id * BIN_SIZE
        BIN_CENTER = BIN_START + (VOTING_BIN_WIDTH / 2)
        predicted_angle = angle_bin_center_offset
        predicted_angles[bin_id] = (predicted_angle + BIN_CENTER) % TAU

    # get rid of the extreme errors on theshold
    bad_bin_ids = []
    for bin_id in range(NUM_OF_VOTING_BINS):
        my_predicted_angle = predicted_angles[bin_id]
        other_bin_ids = list(range(NUM_OF_VOTING_BINS))
        other_bin_ids.remove(bin_id)
        deltas = [
            abs(my_predicted_angle - predicted_angles[other_bin_id])
            for other_bin_id in other_bin_ids
        ]
        delta_threshold = max(deltas) - min(deltas)
        condition_above_thresholds_from_other_bins = [
            delta > delta_threshold for delta in deltas
        ]
        if (
            sum(condition_above_thresholds_from_other_bins) >= NUM_OF_VOTING_BINS - 1
            and delta_threshold < VOTING_BIN_THRESHOLD
        ):
            bad_bin_ids.append(bin_id)
    good_predicted_angles = np.delete(predicted_angles, bad_bin_ids)

    # compute the weighted average
    mean_angle = np.average(good_predicted_angles)
    return mean_angle


# multi affinity bin
NUM_OF_MULTI_AFFINITY_BIN = 2
SHAPE_MULTI_AFFINITY_BIN = (NUM_OF_MULTI_AFFINITY_BIN, 3)
MULTI_AFFINITY_BIN_WIDTH = TAU / NUM_OF_MULTI_AFFINITY_BIN


def radians_to_multi_affinity_bin(angle_rad):
    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU

    # placeholder for all output values
    orientation = np.zeros(shape=SHAPE_MULTI_AFFINITY_BIN)

    for bin_id in range(NUM_OF_MULTI_AFFINITY_BIN):
        # BIN START
        BIN_START = bin_id * BIN_SIZE
        BIN_CENTER = BIN_START + (MULTI_AFFINITY_BIN_WIDTH / 2)
        angle_bin_start_offset = new_angle_rad - BIN_CENTER
        # calculate bin affinity with cos and sin
        orientation[bin_id][:2] = radians_to_single_bin(angle_bin_start_offset)
        # get the confidence (equal distribution)
        orientation[bin_id][2] = np.asarray([1.0]) / NUM_OF_MULTI_AFFINITY_BIN

    # print(f'return shape:\n {orientation[:,:2].shape}\n{orientation[:,2, np.newaxis].shape}')
    return orientation[:, :2], orientation[:, 2, np.newaxis]


def multi_affinity_to_radians(orientation):

    # get the angles and get confidence
    predicted_angles_without_offset = orientation[:, :2]
    predicted_confs = orientation[:, 2]

    # offset each predicted angle by bin center
    predicted_angles = np.empty(predicted_angles_without_offset.shape)
    for bin_id in range(NUM_OF_MULTI_AFFINITY_BIN):
        # get the angle
        # angle_bin_center_offset = single_bin_to_radians(orientation[bin_id,:2])
        BIN_START = bin_id * BIN_SIZE
        BIN_CENTER = BIN_START + (MULTI_AFFINITY_BIN_WIDTH / 2)
        BIN_CENTER_COORD = radians_to_single_bin(BIN_CENTER)
        predicted_angles[bin_id] = predicted_angles_without_offset[bin_id] + BIN_CENTER_COORD

    # compute the weighted average sin and cos
    cos_angles, sin_angles = predicted_angles[:, 0], predicted_angles[:, 1]
    weighted_cos = np.average(cos_angles, weights=predicted_confs)
    weighted_sin = np.average(sin_angles, weights=predicted_confs)
    # scale values between -1 and 1 for acos.
    scale_factor = np.max(np.concatenate([cos_angles, sin_angles]))
    weighted_cos, weighted_sin = weighted_cos / scale_factor, weighted_sin / scale_factor
    # get predicted_angle
    predicted_angle = np.arctan2(weighted_sin, weighted_cos)

    return predicted_angle


def batch_multi_affinity_to_radians(batch):
    return tf.map_fn(multi_affinity_to_radians, batch)


def batch_radians_to_multi_affinity_bin(batch):
    return tf.map_fn(radians_to_multi_affinity_bin, batch)

'''
------------ Start of Experiement ------------
'''

# Experiment A:2-bins, with (cos, sin) pair encoding, with simple average, global angle
EXP_A_NUM_BIN = 6
SHAPE_EXP_A = (EXP_A_NUM_BIN, 2)
EXP_A_BIN_WIDTH = TAU / EXP_A_NUM_BIN

# Following Voting Bin
def radians_to_expA(angle_rad):
    # convert all angles regardless of sign to the range [0-tau)
    new_angle_rad = angle_rad % TAU
    # placeholder for all output values
    orientation = np.zeros(shape=SHAPE_EXP_A)

    for bin_id in range(EXP_A_NUM_BIN):
        # BIN START
        BIN_START = bin_id * EXP_A_BIN_WIDTH
        BIN_CENTER = BIN_START + (EXP_A_BIN_WIDTH / 2)
        angle_bin_center_offset = new_angle_rad - BIN_CENTER
        # calculate bin affinity with cos and sin
        orientation[bin_id] = radians_to_single_bin(angle_bin_center_offset)
    
    return orientation

def expA_to_radians(orientation):
    # clip values between -1 and 1 for acos.
    orientation = np.clip(orientation, -1.0, 1.0)

    # placeholder for the list of all predictions
    predicted_angles = np.empty(shape=(EXP_A_NUM_BIN,))

    # get the weighted average of all predictions
    for bin_id in range(EXP_A_NUM_BIN):
        # get the angle
        angle_bin_center_offset = single_bin_to_radians(orientation[bin_id])
        BIN_START = bin_id * BIN_SIZE
        BIN_CENTER = BIN_START + (VOTING_BIN_WIDTH / 2)
        predicted_angle = angle_bin_center_offset
        predicted_angles[bin_id] = (predicted_angle + BIN_CENTER) % TAU
    # get the average of cos and sin across two bins
    mean_angle = get_mean_angle(predicted_angles)
    return mean_angle

def get_output_shape_dict():
    return {
        "rot-y": SHAPE_ALPHA_ROT_Y,
        "alpha": SHAPE_ALPHA_ROT_Y,
        "multibin":SHAPE_MULTI_AFFINITY_BIN,
        'voting-bin':SHAPE_VOTING_BIN,
        'single-bin': SHAPE_SINGLE_BIN,
        'tricosine' : SHAPE_TRICOSINE,
        'exp-A':SHAPE_EXP_A
    }
