import pdb

import torch
import logging
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import numpy as np
from models.util import weight_reduce_loss
from torch.autograd import Variable
# from .loss.seg_loss import ExpLog_loss
from models.losses import LMFLoss
binary_cross_entropy = F.binary_cross_entropy
#idh_cross_entropy = F.cross_entropy
grade_cross_entropy = F.cross_entropy

from torch import nn


# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num,loss_fn):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.loss_fn = loss_fn
#         self.log_vars = nn.Parameter(torch.tensor((5.0,6.0,6.0),requires_grad=True)) #1.0, 6.0 #6.5,7.0,7.0
#
#     def forward(self, outputs,targets,weights):
#         std_2 = torch.exp(self.log_vars[0]) ** 0.5
#         std_3=torch.exp(self.log_vars[1])**0.5
#         std_4=torch.exp(self.log_vars[2])**0.5
#
#         idh_loss = self.loss_fn[0](outputs[0], targets[0],weights[0])
#         idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * idh_loss + self.log_vars[0],-1)
#
#         atrx_loss = self.loss_fn[1](outputs[1], targets[1],weights[1])
#         atrx_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * atrx_loss + self.log_vars[1], -1)
#
#         p19q_loss = self.loss_fn[2](outputs[2], targets[2],weights[2])
#         p19q_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[2]) * p19q_loss + self.log_vars[2], -1)
#
#         loss = torch.mean(idh_loss_1+atrx_loss_1+p19q_loss_1)
#
#         return loss,idh_loss,atrx_loss,p19q_loss,std_2,std_3,std_4,self.log_vars[0],self.log_vars[1],self.log_vars[2]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num,loss_fn):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        self.log_vars = nn.Parameter(torch.tensor((5.0,6.0,6.0),requires_grad=True)) #1.0, 6.0 #6.5,7.0,7.0

    def forward(self, outputs,targets,weights):
        std_2 = torch.exp(self.log_vars[0]) ** 0.5
        std_3=torch.exp(self.log_vars[1])**0.5
        std_4=torch.exp(self.log_vars[2])**0.5
        if(weights[0] is None):
            idh_loss = self.loss_fn[0](outputs[0], targets[0], weights[0])
            idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * idh_loss + self.log_vars[0], -1)

            atrx_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])
            atrx_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * atrx_loss + self.log_vars[1], -1)

            p19q_loss = self.loss_fn[2](outputs[2], targets[2], weights[2])
            p19q_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[2]) * p19q_loss + self.log_vars[2], -1)
        else:
            idh_loss = self.loss_fn[0](outputs[0], targets[0],weights[0].to(outputs[0].device))
            idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * idh_loss + self.log_vars[0],-1)

            atrx_loss = self.loss_fn[1](outputs[1], targets[1],weights[1].to(outputs[1].device))
            atrx_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * atrx_loss + self.log_vars[1], -1)

            p19q_loss = self.loss_fn[2](outputs[2], targets[2],weights[2].to(outputs[2].device))
            p19q_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[2]) * p19q_loss + self.log_vars[2], -1)

        loss = torch.mean(idh_loss_1+atrx_loss_1+p19q_loss_1)

        return loss,idh_loss,atrx_loss,p19q_loss,std_2,std_3,std_4,self.log_vars[0],self.log_vars[1],self.log_vars[2]

def idh_cross_entropy(input,target,weight):
    return cross_entropy(input,target,weight=weight,ignore_index=-1)
def atrx_cross_entropy(input,target,weight):
    return cross_entropy(input,target,weight=weight,ignore_index=-1)
def p19q_cross_entropy(input,target,weight):
    return cross_entropy(input,target,weight=weight,ignore_index=-1)

def idh_focal_loss(input,target):
    return focal_loss(input,target)
def atrx_focal_loss(input,target):
    return focal_loss(input,target)
def p19q_focal_loss(input,target):
    return focal_loss(input,target)

def focal_loss( input, target,alpha=0.25,gamma=2):
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    pt = torch.exp(-ce_loss(input, target))
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss(input, target)
    return torch.mean(focal_loss)

def idh_lmfloss(input,target,weight):
    cls_num_list=[66,54]
    weight=torch.tensor([54, 66]).float()
    device=torch.device("cuda")
    lmf_loss=LMFLoss(cls_num_list, weight.to(device) if weight is not None else None, alpha=1, beta=1, gamma=2, max_m=0.5, s=30)
    return lmf_loss(input.to(device), target.to(device))
    # lmf_loss = LMFLoss(cls_num_list, weight if weight is not None else None, alpha=1, beta=1, gamma=2,
    #                    max_m=0.5, s=30)
    #return lmf_loss(input, target)

def atrx_lmfloss(input,target,weight):
    cls_num_list=[90,30]
    weight=torch.tensor([30, 90]).float()
    device = torch.device("cuda")
    lmf_loss=LMFLoss(cls_num_list, weight.to(device) if weight is not None else None, alpha=1, beta=1, gamma=2, max_m=0.5, s=30)
    return lmf_loss(input.to(device),target.to(device))
    # lmf_loss = LMFLoss(cls_num_list, weight if weight is not None else None, alpha=1, beta=1, gamma=2,
    #                    max_m=0.5, s=30)
    # return lmf_loss(input, target)

def p19q_lmfloss(input,target,weight):
    cls_num_list=[107,13]
    weight=torch.tensor([20,120]).float()
    device = torch.device("cuda")
    lmf_loss=LMFLoss(cls_num_list, weight.to(device) if weight is not None else None, alpha=1, beta=1, gamma=2, max_m=0.5, s=30)
    return lmf_loss(input.to(device),target.to(device))
    # lmf_loss = LMFLoss(cls_num_list, weight if weight is not None else None, alpha=1, beta=1, gamma=2,
    #                    max_m=0.5, s=30)
    # return lmf_loss(input, target)

# def focal_loss(input,target,weight=None,gamma=2.0):
#     input = input.float()
#     target = target.long()
#     ce_loss = F.cross_entropy(input, target, weight=weight, ignore_index=-1)
#     pt = torch.exp(-ce_loss)
#     focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
#     return focal_loss

def build_masked_loss(output, target, weight=None,mask_value=-1):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    return cross_entropy(output, target, weight=weight,ignore_index=mask_value)


def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)





def get_boundary_3d(gtmasks):
    laplacian_kernel = torch.tensor(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 26, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        dtype=torch.float32, device=gtmasks.device).reshape(1, 1, 3, 3, 3).requires_grad_(False)
    boundary_targets = F.conv3d(gtmasks.unsqueeze(1), laplacian_kernel, padding=1)
    boundary_targets = boundary_targets.clamp(min=0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets

class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

