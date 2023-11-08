import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
#from torchviz import make_dot

class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target, smooth=1e-7):
        prediction = prediction.view(-1)
        target = target.view(-1)

        intersection = (prediction * target).sum()
        dice = (2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth)

        return 1 - dice