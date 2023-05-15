import numpy as np
import tensorflow as tf

from ai4med.components.losses.loss import Loss


class BoundaryAndDice(Loss):

    def __init__(self, **args):
        Loss.__init__(self)
        self.args = args

    def get_loss(self, predictions, targets, build_ctx=None):
        return self.boundary_loss(predictions, targets, **self.args)

    def boundary_loss(self,
                      predictions,
                      targets,
                      data_format='channels_first',
                      skip_background=False,
                      squared_pred=False,
                      jaccard=False,
                      smooth=1e-5,
                      top_smooth=0.0,
                      is_onehot_targets=False,
                      alpha=None,
                      boundary_weight=1.0):
        """
        Compute average boundary loss between two 5D tensors (for 3D images)
        :param targets: Tensor of True segmentation values. Usually has 1 channel dimension (e.g. Nx1xHxWxD),
                        where each element is an index indicating class label.
                        Alternatively it can be a one-hot-encoded tensor of the shape NxCxHxWxD, where each channel is  binary (or float in interval 0..1) indicating
                        the probability of the corresponding class label (in this case you must set is_onehot_targets == True)
        :param predictions: Tensor of Predicted segmentation output (e.g NxCxHxWxD) all values are assumed between 0..1
        :param data_format: data format: channels_first (default) or channels_last
        :param skip_background: skip dice computation on the first channel of the predicted output (skipping dice on background class)
        :param squared_pred: bool (default Fals) use squared versions of targets and predictions in the denominator
        :param jaccard: bool (default False) compute Jaccard Index (soft IoU) instead of dice
        :param smooth: denominator constant to avoid zero division (default 1e-5)
        :param top_smooth: (default 0) experimental, nominator constant to avoid zero final loss when targets are all zeros
        :param is_onehot_targets: bool (default False) indicating if the targets are already one-hot-encoded

        :returns float tensor (one minus average dice) the value is 0..1 bound
        """

        is_channels_first = (data_format == 'channels_first')
        ch_axis = 1 if is_channels_first else -1

        n_channels_pred = predictions.get_shape()[ch_axis].value
        n_channels_targ = targets.get_shape()[ch_axis].value
        n_len = len(predictions.get_shape())

        print('dice_loss targets', targets.get_shape().as_list(), 'predictions', predictions.get_shape().as_list(), 'targets.dtype', targets.dtype, 'predictions.dtype', predictions.dtype)
        print('dice_loss is_channels_first:', is_channels_first,'skip_background:', skip_background, 'is_onehot_targets', is_onehot_targets)

        # Sanity checks
        if skip_background and n_channels_pred==1:
            raise ValueError("There is only 1 single channel in the predicted output, and skip_zero is True")
        if skip_background and n_channels_targ==1 and is_onehot_targets:
            raise ValueError("There is only 1 single channel in the true output (and it is is_onehot_true), and skip_zero is True")
        if is_onehot_targets and n_channels_targ!=n_channels_pred:
            raise ValueError("Number of channels in target {} and pred outputs {} must be equal to use is_onehot_true == True".format(
                n_channels_targ, n_channels_pred
            ))
        # End sanity checks

        if not is_onehot_targets:
            # if not one-hot representation already
            targets = tf.cast(tf.squeeze(targets, axis=ch_axis), tf.int32)  # remove singleton (channel) dimension for true labels
            targets = tf.one_hot(targets, depth = n_channels_pred, axis=ch_axis, dtype=tf.float32, name="loss_dice_targets_onehot")

        if skip_background:
            # if skipping background, removing first channel
            targets = targets[:, 1:] if is_channels_first else targets[...,1:]
            predictions = predictions[:, 1:] if is_channels_first else predictions[...,1:]

        # boundary loss
        f1 = 0
        for i in range(n_channels_pred):
            if i == 0 and skip_background:
                continue
            f0 = tf.square(self.compute_boundary(predictions[:, i:i+1, ...]) - self.compute_boundary(targets[:, i:i+1, ...]))
            f1 += tf.reduce_mean(f0)
            # f1 += tf.nn.l2_loss(self.compute_boundary(predictions[:, i:i+1, ...])- self.compute_boundary(targets[:, i:i+1, ...]))
        f1 = f1 / (n_channels_pred - 1) if skip_background else f1 / n_channels_pred

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, n_len)) if is_channels_first else list(range(1, n_len - 1))

        intersection = tf.reduce_sum(targets * predictions, axis=reduce_axis)

        if squared_pred:
            targets = tf.square(targets)  # technically we don't need this square for binary true values (but in cases where true is probability/float, we still need to square
            predictions = tf.square(predictions)

        y_true_o = tf.reduce_sum(targets, axis=reduce_axis)
        y_pred_o = tf.reduce_sum(predictions, axis=reduce_axis)

        denominator = y_true_o + y_pred_o

        if jaccard:
            denominator -= intersection

        f = (2.0 * intersection + top_smooth) / (denominator + smooth)
        f = 1.0 - tf.reduce_mean(f)

        return f + boundary_weight*f1

    def compute_boundary(self, tf_nda):

        # processing
        W = np.zeros(shape=(1, 1, 3, 3, 3), dtype=np.float32)
        W[..., 0, :, :] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        W[..., 1, :, :] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]], dtype=np.float32)
        W[..., 2, :, :] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

        k_size = 3
        y = tf.layers.conv3d(tf_nda,
                             filters=1,
                             kernel_size=k_size,
                             padding='same',
                             data_format='channels_first',
                             use_bias=False,
                             kernel_initializer=tf.constant_initializer(1.0 / float(k_size ** 3)),
                             trainable=False)
        y = tf.layers.conv3d(y,
                             filters=1,
                             kernel_size=k_size,
                             padding='same',
                             data_format='channels_first',
                             use_bias=False,
                             kernel_initializer=tf.constant_initializer(W),
                             trainable=False)

        return y
