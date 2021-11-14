import tensorflow as tf
from orientation_converters import *
from orientation_converters import SHAPE_TRICOSINE, SHAPE_SINGLE_BIN, SHAPE_VOTING_BIN


TF_TYPE = tf.dtypes.float32

# Stateful metric over the entire dataset.
# Because metrics are evaluated for each batch during training and evaluation,
# this metric will keep track of average accuracy over the entire dataset,
# not the average accuracy of each batch.
class OrientationAccuracy(tf.keras.metrics.Metric):

    # Create the state variables in __init__
    def __init__(self, orientation_type, name='orientation_accuracy', **kwargs):
        super(OrientationAccuracy, self).__init__(name=name, **kwargs)
        # internal state variables
        self.orientation_type = orientation_type
        self.reset_states()

    def sum_angle_accuracy(self, alpha_true, alpha_pred):
        alpha_delta = alpha_true - alpha_pred
        orientation_accuracies = 0.5 * (tf.math.cos(alpha_delta) + 1.0)
        return tf.math.reduce_sum(orientation_accuracies)

    def convert_to_radians(self, tensor):
        # if orientation type is already 'alpha' or 'rot_y', no need to change
        if self.orientation_type in ['rot-y', 'alpha']:
            return angle_normed_to_radians(tensor)
        elif self.orientation_type == 'multibin':
            # return batch_multibin_to_batch_radians(tensor)
            return batch_multi_affinity_to_radians(tensor)
        else:
            return self.recursively_convert_to_radians(tensor)

    @tf.autograph.experimental.do_not_convert
    def recursively_convert_to_radians(self, tensor):
        # recursively unpacks tensor until the tensor dimension is same shape as orientation_converters
        tensor_shape = tensor.get_shape()
        arr = tensor.numpy()
        if self.orientation_type == 'tricosine':
            if tensor_shape == SHAPE_TRICOSINE:
                alpha = tricosine_to_radians(arr)
                return tf.constant(alpha, dtype=TF_TYPE)
            elif len(tensor_shape) > len(SHAPE_TRICOSINE):
                return tf.stack([self.recursively_convert_to_radians(un_packed_tensor)
                                 for un_packed_tensor in tf.unstack(tensor)])
        # elif self.orientation_type == 'multibin':
        #     if tensor_shape == SHAPE_MULTIBIN:
        #         radians = multibin_orientation_confidence_to_radians(tensor[..., :2], tensor[..., 2:])
        #         return tf.constant(radians, dtype=TF_TYPE)
        #     elif len(tensor_shape) > len(SHAPE_MULTIBIN):
        #         return tf.stack([self.recursively_convert_to_radians(un_packed_tensor)
        #                          for un_packed_tensor in tf.unstack(tensor)])
        elif self.orientation_type == 'voting-bin':
            if tensor_shape == SHAPE_VOTING_BIN:
                alpha = voting_bin_to_radians(arr)
                return tf.constant(alpha, dtype=TF_TYPE)
            elif len(tensor_shape) > len(SHAPE_VOTING_BIN):
                return tf.stack([self.recursively_convert_to_radians(un_packed_tensor)
                                 for un_packed_tensor in tf.unstack(tensor)])
        elif self.orientation_type == 'single-bin':
            if tensor_shape == SHAPE_SINGLE_BIN:
                alpha = single_bin_to_radians(arr)
                return tf.constant(alpha, dtype=TF_TYPE)
            elif len(tensor_shape) > len(SHAPE_SINGLE_BIN):
                return tf.stack([self.recursively_convert_to_radians(un_packed_tensor)
                                 for un_packed_tensor in tf.unstack(tensor)])
        else:
            raise Exception("Invalid self.orientation_type: " +
                            self.orientation_type)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update the variables given y_true and y_pred in update_state()
        # convert to alphas using orientation_converters and calculate the batch_accuracies
        alpha_true = self.convert_to_radians(y_true)
        alpha_pred = self.convert_to_radians(y_pred)
        batch_sum_accuracy = tf.cast(self.sum_angle_accuracy(alpha_true, alpha_pred), tf.float32)

        # update the cur_accuracy
        self.sum_accuracy.assign_add(batch_sum_accuracy)
        self.num_pairs.assign_add(y_pred.get_shape()[0])

    # Return the metric result in result()
    def result(self):
        return tf.math.divide(self.sum_accuracy, tf.cast(self.num_pairs, dtype=TF_TYPE))

    # Reset state
    def reset_state(self):
        self.num_pairs = tf.Variable(0, dtype=tf.dtypes.int32)  # num of pairs of y_true, y_pred
        # sum of accuracies for each pair of y_true, y_pred
        self.sum_accuracy = tf.Variable(0., dtype=TF_TYPE)
    