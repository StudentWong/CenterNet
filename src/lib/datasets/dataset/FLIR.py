from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class FLIR(data.Dataset):
  num_classes = 3
  default_resolution = [384, 384]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(FLIR, self).__init__()
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
    coco_eval.params.catIds = [1, 2, 3]
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