from monai.losses import DiceLoss

import torch
from torch.nn.modules.loss import _Loss

#import importlib
#boundaryloss = importlib.import_module("boundary-loss")
from boundaryloss.losses import BoundaryLoss

class HybridDiceLoss(DiceLoss):

    #def __init__(self, include_background: bool = True, to_onehot_y: bool = False, sigmoid: bool = False, softmax: bool = False, one_act = Optional[Callable] = None, square_pred: bool = False, jaccard: bool = False, reduction: Union ):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boundary_loss = BoundaryLoss(idc=[1])
        self.dice_loss = super().forward
        self.alpha = 0.01
        #self.dice_loss = DiceLoss(

    def forward(self, input: torch.Tensor, target: torch.Tensor, dist_map: torch.Tensor) -> torch.Tensor:
        d_loss = self.dice_loss(input, target)
        b_loss = self.boundary_loss(input, dist_map)
        return self.d_loss + self.alpha * self.b_loss
