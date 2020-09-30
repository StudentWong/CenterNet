from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class ShangQi(data.Dataset):
  num_classes = 5
  default_resolution = [384, 384]
  mean = np.array([0.317200417, 0.317200417, 0.317200417],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.22074733, 0.22074733, 0.22074733],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(ShangQi, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'shangqi')
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
    self.class_name = [
      '__background__', 'car', 'person', 'rider', 'truck', 'bus']
    self._valid_ids = [
      1, 2, 3, 7, 8]


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

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

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
    coco_eval.params.catIds = [1, 2, 3, 7, 8]
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

def shangqi_cal_mean_and_std(train_data_dir='/home/studentw/disk3/tracker/CenterNet/data/shangqi/train', val_data_dir=None):
  train_list = sorted(os.listdir(train_data_dir))
  all_img=[]
  for img_name in train_list:
    all_img = all_img + [os.path.join(train_data_dir, img_name)]
  if not val_data_dir is None:
    val_list = sorted(os.listdir(val_data_dir))
    for img_name in val_list:
      all_img = all_img + [os.path.join(val_data_dir, img_name)]

  R_channel_mean = 0
  G_channel_mean = 0
  B_channel_mean = 0
  width=None
  height=None
  for img_path in all_img:
    img = cv2.imread(img_path)/255.0
    if width is None:
      width = img.shape[0]
    else:
      assert width == img.shape[0], 'img shape error'
    if height is None:
      height = img.shape[1]
    else:
      assert height == img.shape[1], 'img shape error'
    R_channel_mean = R_channel_mean + np.sum(img[:, :, 0])
    G_channel_mean = G_channel_mean + np.sum(img[:, :, 1])
    B_channel_mean = B_channel_mean + np.sum(img[:, :, 2])

  num = len(all_img) * width * height  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
  R_mean = R_channel_mean / num
  G_mean = G_channel_mean / num
  B_mean = B_channel_mean / num

  print(R_mean)
  print(G_mean)
  print(B_mean)

  R_channel_std = 0
  G_channel_std = 0
  B_channel_std = 0
  for img_path in all_img:
    img = cv2.imread(img_path)/255.0
    R_channel_std = R_channel_std + np.sum((img[:, :, 0] - R_mean) ** 2)
    G_channel_std = G_channel_std + np.sum((img[:, :, 1] - G_mean) ** 2)
    B_channel_std = B_channel_std + np.sum((img[:, :, 2] - B_mean) ** 2)

  R_std = np.sqrt(R_channel_std / num)
  G_std = np.sqrt(G_channel_std / num)
  B_std = np.sqrt(B_channel_std / num)
  print(R_std)
  print(G_std)
  print(B_std)

    # print(type(img))

  # print(img_list)

def save_rect_img():
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

def eval_result():
  shangqi_coco = coco.COCO('/home/studentw/disk3/tracker/CenterNet/data/shangqi/annotations/val.json')
  shangqi_coco_dets = shangqi_coco.loadRes('/home/studentw/disk3/tracker/CenterNet/exp/ctdetnfs/default/results.json')

  # print(shangqi_coco_dets[imgids[0]])
  coco_eval = COCOeval(shangqi_coco, shangqi_coco_dets, "bbox")
  coco_eval.params.catIds = [1,2,3,7,8]
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
  # print(ret)


import json
import cv2
import os
if __name__ == '__main__':
  eval_result()