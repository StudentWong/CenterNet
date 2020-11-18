from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.pose_dla_dcn_no_bias import get_pose_net_no_bias as get_dla_dcn_no_bias
from .networks.pose_dla_dcn_freeze_update import get_pose_net_freeze_update as get_dla_dcn_freeze_update
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.resnet_dcn_4c import get_pose_net as get_pose_net_dcn_4c
from .networks.large_hourglass import get_large_hourglass_net
from .networks.pose_dla_dcn_gt_centerloss import get_pose_net_gt_centerloss as get_dla_dcn_gt_centerloss
from .networks.resnet_dcn_gt import get_pose_net as get_res_gt_net_dcn
from .networks.resnet_dcn_gt_4c import get_pose_net as get_res_gt_net_dcn_4c
from .networks.resnet_dcn_4c_share import get_pose_net as get_res_gt_net_dcn_4c_share

_model_factory = {
  'res': get_pose_net, # default Resnet with deconv
  'dlav0': get_dlav0, # default DLAup
  'dla': get_dla_dcn,
  'dlaNoBias': get_dla_dcn_no_bias,
  'dlaFreezeUpdate': get_dla_dcn_freeze_update,
  'dlagt': get_dla_dcn_gt_centerloss,
  'resdcn': get_pose_net_dcn,
  'resdcn4c': get_pose_net_dcn_4c,
  'resdcn4cshare': get_res_gt_net_dcn_4c_share,
  'resdcngt': get_res_gt_net_dcn,
  'resdcngt4c': get_res_gt_net_dcn_4c,
  'hourglass': get_large_hourglass_net,
}

def create_model(arch, heads, head_conv, adapt_thermal_weight=0.5):
  num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
  arch = arch[:arch.find('_')] if '_' in arch else arch
  get_model = _model_factory[arch]
  # print(num_layers)
  # print(heads)
  # print(head_conv)
  model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv, adapt_thermal_weight=adapt_thermal_weight)
  return model

def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None, lamda_augment=1.0):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  # print(model_state_dict.keys())
  # exit()
  for k in state_dict:
    # if k == 'menber_activation.lamda':
    #   print(state_dict[k])
    #   exit()
    #   state_dict[k] = lamda_augment * state_dict[k]
    # if k == 'menber_activation.c':
    #   for i in range(0, state_dict[k].shape[0]):
    #     print(state_dict[k][i])
    #   exit()
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def load_model_freeze(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  if 'lamda' in checkpoint.keys():
    lamda = checkpoint['lamda']
    model.member_lamda = lamda.cuda()
  if 'center' in checkpoint.keys():
    center = checkpoint['center']
    model.member_c = center.cuda()

  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, ' \
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model

def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

def save_model_freeze(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict,
          'center': model.member_c,
          'lamda': model.member_lamda}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()

  torch.save(data, path)


