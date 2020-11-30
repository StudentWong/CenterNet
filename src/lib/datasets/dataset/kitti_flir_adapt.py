from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
# print(os.getcwd())
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
import torch.utils.data as data
from .FLIR_adapt import FLIR_adapt
from .kitti_adapt import KITTI_adapt

class KITTI_FLIR_adapt(data.Dataset):
  num_classes = 3
  default_resolution = [192, 192]

  def get_object_list(self, dataset, idx):
    img_id = dataset.images[idx]
    ann_ids = dataset.coco.getAnnIds(imgIds=[img_id])
    anns = dataset.coco.loadAnns(ids=ann_ids)
    cat_list = []
    for ann in anns:
      if ann['category_id'] not in cat_list:
        cat_list = cat_list + [ann['category_id']]
    return cat_list

  def __init__(self, opt, split):
    self.flir_data = FLIR_adapt(opt, split)
    self.kitti_data = KITTI_adapt(opt, split, [512, 640])
    self.len_flir = len(self.flir_data)
    self.len_kitti = len(self.kitti_data)
    self.kitti_remain = list(range(0, self.len_kitti))

    self.flir_classify = {None: [], 1: [], 2: [], 3: []}

    for i in range(self.len_flir):
      obj_list = self.get_object_list(self.flir_data, i)
      # img_id = flirdata.images[i]
      if len(obj_list) == 0:
        self.flir_classify[None] = self.flir_classify[None] + [i]
      elif 1 in obj_list:
        self.flir_classify[1] = self.flir_classify[1] + [i]
      elif 2 in obj_list:
        self.flir_classify[2] = self.flir_classify[2] + [i]
      elif 3 in obj_list:
        self.flir_classify[3] = self.flir_classify[3] + [i]

    self.kitti_classify = {None: [], 1: [], 2: [], 3: []}

    for i in range(self.len_kitti):
      obj_list = self.get_object_list(self.kitti_data, i)
      # img_id = flirdata.images[i]
      if len(obj_list) == 0:
        self.kitti_classify[None] = self.kitti_classify[None] + [i]
      elif 1 in obj_list:
        self.kitti_classify[1] = self.kitti_classify[1] + [i]
      elif 2 in obj_list:
        self.kitti_classify[2] = self.kitti_classify[2] + [i]
      elif 3 in obj_list:
        self.kitti_classify[3] = self.kitti_classify[3] + [i]
    # print(self.flir_classify)


  def __len__(self):
    return self.len_flir

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    self.flir_data.save_results(results, save_dir)

  def run_eval(self, results, save_dir):
    self.flir_data.run_eval(results, save_dir)

  def run_eval_return(self, results, save_dir):
    return self.flir_data.run_eval_return(results, save_dir)

# if __name__ == "__main__":
#   annot_path = "/home/htz/caijihuzhuo/CenterNet/data/kitti/train/kitti_train.json"
#   img_dir = os.path.join('/home/htz/caijihuzhuo/CenterNet/data/kitti', 'train')
#   # annot_path = "/home/htz/caijihuzhuo/CenterNet/data/FLIR_ADAS_1_3/train/thermal_annotations.json"
#   # img_dir = os.path.join('/home/htz/caijihuzhuo/CenterNet/data/FLIR_ADAS_1_3', 'train')
#   coco = coco.COCO(annot_path)
#   images = coco.getImgIds()
#   img_id = images[2]
#   file_name = coco.loadImgs(ids=[img_id])[0]['file_name']
#
#   img_path = os.path.join(img_dir, file_name)
#   ann_ids = coco.getAnnIds(imgIds=[img_id])
#   anns = coco.loadAnns(ids=ann_ids)
#   # print(anns)
#   img = cv2.imread(img_path)
#   print(img_path)
#   assert not img is None
#   for ann in anns:
#     # pt1 = (int(ann['bbox'][0]-0.5*ann['bbox'][2]), int(ann['bbox'][1]-0.5*ann['bbox'][3]))
#     # pt2 = (int(ann['bbox'][0] + 0.5 * ann['bbox'][2]), int(ann['bbox'][1] + 0.5 * ann['bbox'][3]))
#     pt1 = (int(ann['bbox'][0]), int(ann['bbox'][1]))
#     pt2 = (int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3]))
#
#     img = cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=2)
#   cv2.imshow("win", img)
#   cv2.waitKey(0)
