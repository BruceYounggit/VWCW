import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


#
# class GaussianMSE(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
# def Gaussian_MSE(x, target, kernel):
#     target = target_transform(x, target, kernel)
#     return F.mse_loss(x, target)


def target_transform(x, target, kernel):
    target = F.adaptive_max_pool2d(target, x.shape[2:])
    with torch.no_grad():
        target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
    return target
