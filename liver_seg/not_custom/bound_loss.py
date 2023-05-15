####
# Chang 20210817
# Modified from Cheng En tensorflow boundary loss code
####

import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from monai.losses import DiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction

class BoundaryAndDice(DiceLoss):
    def __init__(
            self, 
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            boundary_weight: float = 1.0,
            ) -> None:
        super().__init__(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                other_act=other_act,
                squared_pred=squared_pred,
                jaccard=jaccard,
                reduction=reduction,
                smooth_nr=smooth_nr,
                smooth_dr=smooth_dr,
                batch=batch,
                )

        self.boundary_weight = boundary_weight
        self.dice_loss = super()
        self.boundary_loss = BoundaryLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                other_act=other_act,
                reduction=reduction,
                batch=batch,
                boundary_weight=boundary_weight)
                
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: 
        return self.dice_loss.forward(input, target) + self.boundary_weight * self.boundary_loss(input, target)

class BoundaryLoss(_Loss):
    def __init__(
            self,
            include_background: bool = True,
            to_onehot_y: bool = False,
            sigmoid: bool = False,
            softmax: bool = False,
            other_act: Optional[Callable] = None,
            squared_pred: bool = False,
            jaccard: bool = False,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            smooth_nr: float = 1e-5,
            smooth_dr: float = 1e-5,
            batch: bool = False,
            boundary_weight: float = 1.0,
            ) -> None:
                
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warning.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape({target.shape}) from ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        # calculate boundary loss
        f1 = 0
        for i in range(n_pred_ch):
            if i == 0 and not self.include_background:
                continue
            f0 = torch.square(self._compute_boundary(input[:, i:i+1, ...]) - self._compute_boundary(targets[:, i:i+1, ...]))
            f1 += torch.mean(f0)

        if not self.include_background:
            f1 = f1 / (n_pred_ch - 1)
        else:
            f1 / n_pred_ch

        return f1

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
