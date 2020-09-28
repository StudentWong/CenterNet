# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
try:
    from .utils import _transpose_and_gather_feat
except:
    from src.lib.models.utils import _transpose_and_gather_feat
import torch.nn.functional as F

min_clip = 1e-6

def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''

  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  # print(pred)
  # print(gt)
  # print(pred.max())
  # print(gt.max())
  # print(pred.shape)
  # print(gt.shape)
  # exit()
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  # print(torch.log(pred) * torch.pow(1 - pred, 2))
  # print(pos_loss.max())
  # print(pos_loss.min())
  # exit()

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  # print(num_pos)
  # print(pos_loss)
  # print(neg_loss)
  # exit()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss


def distance_no_sqrt(x, y, dim=-1, keepdim=False):
    assert x.shape == y.shape
    shapelen = len(x.shape)
    assert dim < shapelen
    return ((x - y)**2).sum(dim=dim, keepdim=keepdim)

class _menbership_center_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, actviate):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)

        # hidden: N*D*C*(w*h)

        assert x.shape[0] == actviate.shape[0] and len(x.shape) == 3
        N = x.shape[0]
        assert x.shape[1] == c.shape[0] and len(c.shape) == 2
        D = x.shape[1]
        assert c.shape[1] == actviate.shape[1]
        C = c.shape[1]
        assert len(actviate.shape) == 3 and x.shape[2] == actviate.shape[2]
        wh = x.shape[2]

        x_expand = x.unsqueeze(2).expand([N, D, C, wh])
        c_expand = c.unsqueeze(0).unsqueeze(3).expand([N, D, C, wh])
        actviate_expand = actviate.unsqueeze(1).expand([N, D, C, wh])
        distance = distance_no_sqrt(x_expand, c_expand, dim=1)
        ctx.save_for_backward(x_expand.detach(), c_expand.detach(), actviate_expand.detach(), distance.detach())
        #distance shape: N*C*(w*h)
        assert distance.shape[0] == N \
               and distance.shape[1] == C \
               and distance.shape[2] == wh \
               and actviate.shape[0] == N \
               and actviate.shape[1] == C \
               and actviate.shape[2] == wh

        loss_N_C_wh = distance * actviate
        loss = loss_N_C_wh.sum()
        return loss.clamp(min=min_clip)

    @staticmethod
    def backward(ctx, grad_output):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)

        # hidden: N*D*C*(w*h)

        # x_expand: N*D*C*(w*h)
        # c_expand: N*D*C*(w*h)
        # actviate_expand: N*D*C*(w*h)
        # distance: N*C*(w*h)
        x_expand, c_expand, actviate_expand, distance = ctx.saved_variables

        grad_x = grad_c = grad_activate = None

        if ctx.needs_input_grad[0]:
            grad_x_this_layer = (2 * (x_expand - c_expand) * actviate_expand).sum(dim=2)
            grad_x = grad_x_this_layer * grad_output

        if ctx.needs_input_grad[1]:
            grad_c_this_layer = ((2 * (c_expand - x_expand) * actviate_expand).sum(dim=(0, 3))) \
                                / actviate_expand.sum(dim=(0, 3))
            grad_c = grad_c_this_layer * grad_output
        if ctx.needs_input_grad[2]:
            grad_activate_this_layer = distance
            grad_activate = grad_activate_this_layer * grad_output

        return grad_x, grad_c, grad_activate


class _menbership_center_loss_freeze(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, actviate):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)

        # hidden: N*D*C*(w*h)

        assert x.shape[0] == actviate.shape[0] and len(x.shape) == 3
        N = x.shape[0]
        assert x.shape[1] == c.shape[0] and len(c.shape) == 2
        D = x.shape[1]
        assert c.shape[1] == actviate.shape[1]
        C = c.shape[1]
        assert len(actviate.shape) == 3 and x.shape[2] == actviate.shape[2]
        wh = x.shape[2]

        x_expand = x.unsqueeze(2).expand([N, D, C, wh])
        c_expand = c.unsqueeze(0).unsqueeze(3).expand([N, D, C, wh])
        actviate_expand = actviate.unsqueeze(1).expand([N, D, C, wh])
        distance = distance_no_sqrt(x_expand, c_expand, dim=1)
        ctx.save_for_backward(x_expand.detach(), c_expand.detach(), actviate_expand.detach(), distance.detach())
        #distance shape: N*C*(w*h)
        assert distance.shape[0] == N \
               and distance.shape[1] == C \
               and distance.shape[2] == wh \
               and actviate.shape[0] == N \
               and actviate.shape[1] == C \
               and actviate.shape[2] == wh

        loss_N_C_wh = distance * actviate
        loss = loss_N_C_wh.sum()
        return loss.clamp(min=min_clip)

    @staticmethod
    def backward(ctx, grad_output):
        # x: N*D*(w*h)
        # c: D*C
        # activate: N*C*(w*h)

        # hidden: N*D*C*(w*h)

        # x_expand: N*D*C*(w*h)
        # c_expand: N*D*C*(w*h)
        # actviate_expand: N*D*C*(w*h)
        # distance: N*C*(w*h)
        x_expand, c_expand, actviate_expand, distance = ctx.saved_variables

        grad_x = grad_c = grad_activate = None

        if ctx.needs_input_grad[0]:
            grad_x_this_layer = (2 * (x_expand - c_expand) * actviate_expand).sum(dim=2)
            grad_x = grad_x_this_layer * grad_output

        if ctx.needs_input_grad[1]:
            print('warning: c needs grad')
            grad_c_this_layer = ((2 * (c_expand - x_expand) * actviate_expand).sum(dim=(0, 3))) \
                                / actviate_expand.sum(dim=(0, 3))
            grad_c = grad_c_this_layer * grad_output
        if ctx.needs_input_grad[2]:
            grad_activate_this_layer = distance
            grad_activate = grad_activate_this_layer * grad_output

        return grad_x, grad_c, grad_activate

#
class CenterLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(CenterLoss, self).__init__()
    self.center_loss = _menbership_center_loss

  def forward(self, x, c, act):
    return self.center_loss.apply(x, c, act)

class CenterLoss_freeze(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(CenterLoss_freeze, self).__init__()
    self.center_loss = _menbership_center_loss_freeze

  def forward(self, x, c, act):
    return self.center_loss.apply(x, c, act)


class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

# if __name__ == '__main__':
#     a = torch.tensor([[[0, 1, 1], [1, 0, 1]], [[1, 1, 1], [0, 0, 1]],
#                       [[-1, -1, 1], [-1, 0, 1]]], dtype=torch.float, requires_grad=True)
#     b = -a.clone()
#     print(a.shape)
#     # exit()
#     print(distance_no_sqrt(a, b, dim=1))
#     print(distance_no_sqrt(a, b, dim=1).shape)
