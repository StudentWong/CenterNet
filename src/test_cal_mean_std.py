from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from external.nms import soft_nms
from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model, save_model, load_model_freeze, save_model_freeze

def test(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if opt.task != 'ctdetfreeze':
    load_model_fun = load_model
  else:
    load_model_fun = load_model_freeze
  # logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)

  if opt.load_model != '':
    model = load_model_fun(
      model, opt.load_model)
  model.to(device=opt.device)

  # if opt.arch == 'dlaNoBias_34':
  #     optimizer = torch.optim.Adam(model.menber_activation.parameters(), opt.lr)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'val'),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
  )


  train_loader = torch.utils.data.DataLoader(
    Dataset(opt, 'train'),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
  )

  model.eval()
  torch.cuda.empty_cache()

  ft_sum = torch.zeros((64, 20), dtype=torch.float).cuda()
  mask_sum = torch.zeros((64, 20), dtype=torch.float).cuda()

  data_loader = train_loader

  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)

  for iter_id, batch in enumerate(data_loader):
    if iter_id >= num_iters:
      break

    for k in batch:
      if k != 'meta':
        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

    with torch.no_grad():
      # print(batch.keys())
      outputs = model(batch['input'])
      # print(batch.keys())
      # print(outputs[-1].keys())
      # print(batch['reg_mask'].shape)
      # print(batch['ind'].shape)
      mask = batch['hm'].eq(1).float().unsqueeze(1).expand((1, 64, 20, 96, 96)).detach()
      map = (mask * outputs[-1]['ft'].unsqueeze(2).expand((1, 64, 20, 96, 96))).detach()
      # print(mask.sum(dim=(0, 3, 4)))
      # print(map.sum(dim=(0, 3, 4)).detach())
      ft_sum = ft_sum.detach() + map.sum(dim=(0, 3, 4)).detach()
      mask_sum = mask_sum.detach() + mask.sum(dim=(0, 3, 4)).detach()

      # print(outputs[-1]['ft'].max())

    # print(ft_sum)
    # print(mask_sum)
    # mean = ft_sum / mask_sum
    # print(mean)
    # print(model.menber_activation.c)
    # exit()
    print("mean:[{:d}/{:d}]".format(iter_id, num_iters))
    del outputs

  mean = ft_sum/mask_sum
  mean_np = mean.detach().cpu().numpy()
  c_np = model.menber_activation.c.detach().cpu().numpy()
  np.save("./mean_np", mean_np)
  np.save("./c_np", c_np)

  std_sum = torch.zeros((64, 20), dtype=torch.float).cuda()
  mask_std_sum = torch.zeros((64, 20), dtype=torch.float).cuda()
  for iter_id, batch in enumerate(data_loader):
    if iter_id >= num_iters:
      break

    for k in batch:
      if k != 'meta':
        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

    with torch.no_grad():
      # print(batch.keys())
      outputs = model(batch['input'])
      # print(batch.keys())
      # print(outputs[-1].keys())
      # print(batch['reg_mask'].shape)
      # print(batch['ind'].shape)
      mask = batch['hm'].eq(1).float().unsqueeze(1).expand((1, 64, 20, 96, 96)).detach()
      map = (mask * outputs[-1]['ft'].unsqueeze(2).expand((1, 64, 20, 96, 96))).detach()

      mean_unsqueeze = mean.unsqueeze(0).unsqueeze(3).unsqueeze(4).expand_as(mask).detach() * mask.detach()

      # print(mask.sum(dim=(0, 3, 4)))
      # print(map.sum(dim=(0, 3, 4)).detach())
      std_sum = std_sum.detach() + ((map - mean_unsqueeze)**2).sum(dim=(0, 3, 4)).detach()
      mask_std_sum = mask_std_sum.detach() + mask.sum(dim=(0, 3, 4)).detach()

      # print(outputs[-1]['ft'].max())

    # print(ft_sum)
    # print(mask_sum)
    # mean = ft_sum / mask_sum
    # print(mean)
    # print(model.menber_activation.c)
    # exit()
    print("std:[{:d}/{:d}]".format(iter_id, num_iters))
    del outputs

  std = std_sum / mask_std_sum
  std_np = std.detach().cpu().numpy()
  lamda_np = model.menber_activation.lamda.detach().cpu().numpy()
  np.save("./std_np", std_np)
  np.save("./lamda_np", lamda_np)
  # logger.close()

# if __name__ == '__main__':
#   opt = opts().parse()
#
#   test(opt)

if __name__ == '__main__':
  std_np = np.load("./std_np.npy")
  lamda_np = np.load("./lamda_np.npy")
  mean_np = np.load("./mean_np.npy")
  c_np = np.load("./c_np.npy")
  print(mean_np)
  print(c_np)
  print(np.sqrt(std_np))
  print(lamda_np)
