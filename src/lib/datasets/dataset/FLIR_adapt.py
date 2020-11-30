from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
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
import torch.utils.data as data
import math

class FLIR_adapt(data.Dataset):
  num_classes = 3
  # default_resolution = [384, 384]
  # mean = np.array([0.40789654, 0.44719302, 0.47026115],
  #                  dtype=np.float32).reshape(1, 1, 3)
  # std = np.array([0.28863828, 0.27408164, 0.27809835],
  #                  dtype=np.float32).reshape(1, 1, 3)

  mean = np.array([0.44719302, 0.44719302, 0.44719302],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.27809835, 0.27809835, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(FLIR_adapt, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'FLIR_ADAS_1_3')
    self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
    if split == 'test':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'image_info_test-dev2017.json').format(split)
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'instances_extreme_{}2017.json').format(split)
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          '{}.json').format(split)
    self.max_objs = 64
    # cat = {1: 'car', 2: 'person', 3: 'rider', 7: 'truck', 8: 'bus'}
    self.class_name = [
      '__background__', 'car', 'person', 'rider']
    self._valid_ids = [
      3, 1, 2]


    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing FLIR 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    # self.images = []

    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      # print(image_id)
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'), indent=4)
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.params.catIds = self._valid_ids
    coco_eval.params.imgIds = self.images
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

  def run_eval_return(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.params.catIds = self._valid_ids
    coco_eval.params.imgIds = self.images
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    ret = dict()
    ret['AP'] = coco_eval.stats[0]
    ret['AP50'] = coco_eval.stats[1]
    ret['AR1'] = coco_eval.stats[6]
    ret['AR10'] = coco_eval.stats[7]
    ret['AR100'] = coco_eval.stats[8]
    return ret

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
    anns = self.coco.loadAnns(ids=ann_ids)
    # num_objs = min(len(anns), self.max_objs)

    img = cv2.imread(img_path)
    if img is None:
      print(img_path)
    assert not img is None


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


# import json
# import cv2
# import os
# if __name__ == '__main__':
#   shangqi_coco = coco.COCO('/home/studentw/disk3/tracker/CenterNet/data/shangqi/annotations/val.json')
#   shangqi_coco_dets = shangqi_coco.loadRes('/home/studentw/disk3/tracker/CenterNet/exp/ctdet/default/results.json')

  # print(shangqi_coco_dets[imgids[0]])
  # coco_eval = COCOeval(shangqi_coco, shangqi_coco_dets, "bbox")
  # coco_eval.params.catIds = [
  #     8]
  # coco_eval.evaluate()
  # coco_eval.accumulate()
  # coco_eval.summarize()



import json
import cv2
import os
if __name__ == '__main__':
  shangqi_coco = coco.COCO('/home/studentw/disk3/tracker/CenterNet/data/shangqi/annotations/val.json')
  shangqi_coco_dets = shangqi_coco.loadRes('/home/studentw/disk3/tracker/CenterNet/exp/ctdet/default/results.json')

  cat = {1: 'car', 2: 'person', 3: 'rider', 7: 'truck', 8: 'bus'}
  color = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 255, 255), 7: (0, 0, 255), 8: (255, 255, 0)}
  imgroot = '/home/studentw/disk3/tracker/CenterNet/data/shangqi/val'
  imgsave = '/home/studentw/disk3/shangqiresultpic/'
  thre = 0.2
  with open('/home/studentw/disk3/tracker/CenterNet/exp/ctdet/default/results.json', 'r') as fp:
    result = json.load(fp)
  imgids = shangqi_coco_dets.getImgIds()
  # print(len(result))
  visited = np.array([False]*len(result), dtype=np.bool)
  # print(visited)
  # print(len(visited))
  for imgid in imgids:
    targets = []
    for i, det in enumerate(result):
      if visited[i] == True:
        continue
      if det['image_id'] == imgid and det['score']>thre:

        targets = targets + [det]
    # print(targets)
      # print(det)
      # print(imgid)
      # exit()
    img = cv2.imread(os.path.join(imgroot, '{:012d}.png'.format(imgid)))


    for ii in range(0, len(targets)):
      img = cv2.rectangle(img, (int(targets[ii]['bbox'][0]), int(targets[ii]['bbox'][1])),
                                   (int(targets[ii]['bbox'][0] + targets[ii]['bbox'][2]), int(targets[ii]['bbox'][1] + targets[ii]['bbox'][3])), color[targets[ii]['category_id']], thickness=1)
      img = cv2.putText(img, '{}:{:.3f}'.format(cat[targets[ii]['category_id']], targets[ii]['score']),
                  (int(targets[ii]['bbox'][0]), int(targets[ii]['bbox'][1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  color[targets[ii]['category_id']], 1)

    cv2.imwrite(os.path.join(imgsave, '{:012d}.png'.format(imgid)), img)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    #pic =
    # print(len(shangqi_coco_dets.getAnnIds(imgIds=imgid)))
    # print(len(targets))
  # result
  # print(imgids)
