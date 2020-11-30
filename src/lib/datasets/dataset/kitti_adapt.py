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

try:
  from utils.image import flip, color_aug
  from utils.image import get_affine_transform, affine_transform
  from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
  from utils.image import draw_dense_reg
except:
  pass
try:
  from src.lib.utils.image import flip, color_aug
  from src.lib.utils.image import get_affine_transform, affine_transform
  from src.lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
  from src.lib.utils.image import draw_dense_reg
except:
  pass
try:
  from lib.utils.image import flip, color_aug
  from lib.utils.image import get_affine_transform, affine_transform
  from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
  from lib.utils.image import draw_dense_reg
except:
  pass
import torch.utils.data as data

ignore_thre = 0.12
class KITTI_adapt(data.Dataset):
  num_classes = 3
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
  # default_resolution = [384, 1280]
  def __init__(self, opt, split, flir_resolution):
    super(KITTI_adapt, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'kitti')
    # self.img_dir = os.path.join(self.data_dir, 'images', 'trainval')
    self.img_dir = os.path.join(self.data_dir, split)
    self.flir_resolution = flir_resolution
    # self.annot_path = os.path.join(self.data_dir,
    #     'annotations', 'kitti_{}.json').format(split)
    self.annot_path = os.path.join(self.data_dir, split, 'kitti_{}.json'.format(split))
    self.max_objs = 64
    self.class_name = ['__background__', 'car', 'person', 'rider']
    self._valid_ids = [
      3, 1, 2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    self.alpha_in_degree = False

    print('==> initializing kitti {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    pass

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for img_id in results.keys():
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for cls_ind in results[img_id]:
        for j in range(len(results[img_id][cls_ind])):
          class_name = self.class_name[cls_ind]
          f.write('{} 0.0 0'.format(class_name))
          for i in range(len(results[img_id][cls_ind][j])):
            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
          f.write('\n')
      f.close()

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    os.system('./tools/kitti_eval/evaluate_object_3d_offline ' + \
              '../data/kitti/training/label_val ' + \
              '{}/results/'.format(save_dir))

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
      i *= 2
    return border // i

  def get_sample(self, index):
    # print(self._valid_ids)
    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, file_name)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns_o = self.coco.loadAnns(ids=ann_ids)
    # num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    if img is None:
      print(img_path)
    assert not img is None
    # for ann in anns_o:
    #     pt1 = (int(ann['bbox'][0]), int(ann['bbox'][1]))
    #     pt2 = (int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3]))
    #
    #     img = cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=2)
    # cv2.imshow("win", img)
    # cv2.waitKey(0)

    img_shape = img.shape
    scale_h = img_shape[0]/self.flir_resolution[0]
    img = cv2.resize(img,
                     (int(img_shape[1]/scale_h), int(img_shape[0]/scale_h))
                     )
    rand_crop_adaption = np.random.randint(0, img.shape[1]-self.flir_resolution[1])
    border_crop = [rand_crop_adaption, rand_crop_adaption+self.flir_resolution[1]]
    img = img[:, border_crop[0]:border_crop[1], :]
    # cv2.imshow("win", img)
    # print(img.shape)
    # cv2.waitKey(0)
    anns = []
    for ann_o in anns_o:
      for i, box_element in enumerate(ann_o['bbox']):
        ann_o['bbox'][i] = box_element / scale_h
      # top = ann_o['bbox'][1]
      # bottom = ann_o['bbox'][1] + ann_o['bbox'][3]
      left = ann_o['bbox'][0]
      right = ann_o['bbox'][0] + ann_o['bbox'][2]
      if left>border_crop[0] and right<border_crop[1]:
        ann_o['bbox'][0] = ann_o['bbox'][0] - rand_crop_adaption
        anns = anns + [ann_o]
      elif left > border_crop[1] or right < border_crop[0]:
        continue
      elif left>border_crop[0] and right>border_crop[1]:
        iou = (border_crop[1]-left)/(right-left)
        if iou>ignore_thre:
          ann_o['bbox'][0] = ann_o['bbox'][0] - rand_crop_adaption
          ann_o['bbox'][2] = ann_o['bbox'][2] - (right-border_crop[1])-1
          anns = anns + [ann_o]
      elif left<border_crop[0] and right<border_crop[1]:
        iou = (right - border_crop[0]) / (right - left)
        if iou > ignore_thre:
          ann_o['bbox'][0] = 1
          ann_o['bbox'][2] = ann_o['bbox'][2] - (border_crop[0]-left) - 1
          anns = anns + [ann_o]
      # print(ann_o)
    # for ann in anns:
    #     pt1 = (int(ann['bbox'][0]), int(ann['bbox'][1]))
    #     pt2 = (int(ann['bbox'][0] + ann['bbox'][2]), int(ann['bbox'][1] + ann['bbox'][3]))
    #
    #     img = cv2.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=2)
    # cv2.imshow("win", img)
    # cv2.waitKey(0)
    # exit()

    num_objs = min(len(anns), self.max_objs)
    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w

    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] = width - c[0] - 1

    trans_input = get_affine_transform(
      c, s, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input,
                         (input_w, input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
      draw_umich_gaussian

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      invalid = True
      for valid_idx in self._valid_ids:
        if valid_idx == ann['category_id']:
          invalid = False
          break
      if invalid == True:
        continue
      # if ann['bbox'] !=
      bbox = self._coco_box_to_bbox(ann['bbox'])
      # print(ann['category_id'])
      # print(self.cat_ids[6])
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                       ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

    ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
      ret['meta'] = meta
    # cv2.imshow('1', ret['input'].transpose((1,2,0)))
    # cv2.imshow('2', ret['hm'][0:3, :].transpose((1, 2, 0)))
    # cv2.waitKey(0)
    # print(ret['input'].shape)
    # print(ret['hm'].shape)
    # print(ret['ind'].shape)
    # print(ret['wh'].shape)

    return ret

class optt:
  def __init__(self):
    self.data_dir = "/home/htz/caijihuzhuo/CenterNet/data/"
if __name__ == '__main__':
  opt = optt()
  data = KITTI_adapt(opt, 'train', [512, 640])
  data.get_sample(np.random.randint(0, len(data)))
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
