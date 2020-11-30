from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import random
import numpy as np
import torch
import json
import cv2
import os
try:
  from lib.utils.image import flip, color_aug
  from lib.utils.image import get_affine_transform, affine_transform
  from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
  from lib.utils.image import draw_dense_reg
except:
  pass
try:
  from utils.image import flip, color_aug
  from utils.image import get_affine_transform, affine_transform
  from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
  from utils.image import draw_dense_reg
except:
  pass
import math

class CTDetDatasetadapt(data.Dataset):
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    ret_flir = self.flir_data.get_sample(index)

    # ret = dict()
    # ret['flir_input'] = ret_flir['input']
    # ret['flir_hm'] = ret_flir['hm']
    # ret['flir_reg_mask'] = ret_flir['reg_mask']
    # ret['flir_ind'] = ret_flir['ind']
    # ret['flir_wh'] = ret_flir['wh']
    # ret['flir_reg'] = ret_flir['reg']

    flir_have_object = [0, 0, 0]
    if ret_flir['hm'][0, :, :].max() > 0.3:
      flir_have_object[0] = 1
    if ret_flir['hm'][1, :, :].max() > 0.3:
      flir_have_object[1] = 1
    if ret_flir['hm'][2, :, :].max() > 0.3:
      flir_have_object[2] = 1

    if flir_have_object[1] == 1:
      kitti_idx = random.choice(self.kitti_classify[1])
    elif len(self.kitti_remain) != 0:
      kitti_idx = random.choice(self.kitti_remain)
      self.kitti_remain.remove(kitti_idx)
    else:
      self.kitti_remain = list(range(0, self.len_kitti))
    ret_kitti = self.kitti_data.get_sample(kitti_idx)
    # print(ret_flir['reg'].shape)
    # exit()

    # ret['kitti_input'] = ret_kitti['input']
    # ret['kitti_hm'] = ret_kitti['hm']
    # ret['kitti_reg_mask'] = ret_kitti['reg_mask']
    # ret['kitti_ind'] = ret_kitti['ind']
    # ret['kitti_wh'] = ret_kitti['wh']
    # ret['kitti_reg'] = ret_kitti['reg']

    kitti_have_object = [0, 0, 0]
    if ret_kitti['hm'][0, :, :].max() > 0.3:
      kitti_have_object[0] = 1
    if ret_kitti['hm'][1, :, :].max() > 0.3:
      kitti_have_object[1] = 1
    if ret_kitti['hm'][2, :, :].max() > 0.3:
      kitti_have_object[2] = 1
    # print(flir_have_object)
    # print(kitti_have_object)
    # print(ret['flir_hm'].shape)
    # ret['flir_have_object'] = flir_have_object
    # ret['kitti_have_object'] = kitti_have_object

    ret = {'flir_input': ret_flir['input'],
           'flir_hm': ret_flir['hm'],
           'flir_reg_mask': ret_flir['reg_mask'],
           'flir_ind': ret_flir['ind'],
           'flir_wh': ret_flir['wh'],
           'flir_reg': ret_flir['reg'],

           'kitti_input': ret_kitti['input'],
           'kitti_hm': ret_kitti['hm'],
           'kitti_reg_mask': ret_kitti['reg_mask'],
           'kitti_ind': ret_kitti['ind'],
           'kitti_wh': ret_kitti['wh'],
           'kitti_reg': ret_kitti['reg'],

           'flir_have_object': np.array(flir_have_object),
           'kitti_have_object': np.array(kitti_have_object)}
    # imshow = ret_kitti['input']
    # imshow = 255*((imshow.transpose(1, 2, 0) + imshow.min())/(imshow.max()-imshow.min()))
    # hm = ret_kitti['hm']
    # hm = hm.transpose(1, 2, 0) * 255
    # cv2.imshow("hm", hm.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow("win", imshow.astype(np.uint8))
    # cv2.waitKey(0)
    # exit()
    # print(ret_flir['input'])
    # print(ret)
    return ret