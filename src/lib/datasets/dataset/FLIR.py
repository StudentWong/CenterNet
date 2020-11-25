from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

exclude = [1, 5, 17, 20, 23, 37, 38, 42, 43, 64, 67, 71, 75, 76, 92, 93, 96, 123, 228, 234, 248, 268, 318, 337, 346,
           348, 354, 357, 375, 379, 388, 391, 393, 405, 409, 414, 422, 432, 433, 442, 444, 449, 451, 456, 495, 544, 557,
           572, 574, 597, 614, 630, 635, 650, 661, 688, 705, 725, 740, 748, 758, 763, 774, 776, 792, 809, 872, 873, 880,
           900, 902, 906, 917, 924, 925, 926, 941, 943, 948, 967, 978, 990, 1012, 1025, 1033, 1038, 1040, 1058, 1060,
           1065, 1069, 1071, 1073, 1086, 1095, 1097, 1098, 1099, 1105, 1107, 1112, 1114, 1116, 1120, 1121, 1129, 1134,
           1136, 1146, 1160, 1175, 1177, 1191, 1194, 1202, 1211, 1222, 1229, 1238, 1240, 1245, 1253, 1256, 1270, 1275,
           1280, 1296, 1307, 1310, 1316, 1318, 1321, 1324, 1328, 1334, 1345, 1358, 1359, 1363, 1367, 1371, 1375, 1384,
           1386, 1392, 1398, 1400, 1404, 1405, 1407, 1421, 1429, 1433, 1434, 1435, 1437, 1438, 1442, 1445, 1453, 1457,
           1458, 1465, 1468, 1469, 1472, 1482, 1484, 1488, 1490, 1492, 1496, 1498, 1500, 1502, 1504, 1506, 1507, 1517,
           1523, 1537, 1540, 1542, 1544, 1548, 1571, 1573, 1579, 1581, 1583, 1589, 1609, 1624, 1625, 1637, 1639, 1640,
           1646, 1654, 1687, 1688, 1689, 1690, 1691, 1706, 1707, 1708, 1719, 1720, 1721, 1722, 1765, 1785, 1786, 1787,
           1788, 1789, 1790, 1802, 1803, 1804, 1805, 1806, 1865, 1869, 1876, 1885, 1886, 1910, 1911, 1912, 1913, 1914,
           1918, 1919, 1920, 1975, 1976, 1977, 1978, 1982, 2001, 2002, 2003, 2004, 2005, 2006, 2011, 2012, 2024, 2027,
           2034, 2035, 2060, 2061, 2065, 2086, 2087, 2088, 2148, 2149, 2152, 2153, 2210, 2244, 2261, 2262, 2263, 2264,
           2265, 2266, 2267, 2272, 2273, 2283, 2297, 2300, 2301, 2302, 2303, 2304, 2305, 2313, 2314, 2315, 2316, 2331,
           2332, 2358, 2360, 2370, 2373, 2374, 2376, 2384, 2390, 2426, 2427, 2428, 2429, 2430, 2469, 2470, 2471, 2472,
           2473, 2526, 2527, 2528, 2529, 2530, 2628, 2629, 2675, 2676, 2677, 2678, 2679, 2680, 2688, 2695, 2708, 2739,
           2740, 2741, 2742, 2745, 2746, 2747, 2748, 2749, 2752, 2770, 2869, 2870, 2871, 2872, 2920, 2925, 2926, 2932,
           2941, 2989, 2990, 2991, 3009, 3010, 3011, 3012, 3037, 3038, 3043, 3073, 3094, 3095, 3096, 3097, 3098, 3099,
           3117, 3133, 3162, 3165, 3174, 3175, 3176, 3177, 3178, 3199, 3208, 3209, 3210, 3211, 3212, 3213, 3216, 3219,
           3220, 3225, 3226, 3239, 3270, 3271, 3285, 3286, 3294, 3297, 3300, 3303, 3307, 3310, 3311, 3312, 3334, 3335,
           3336, 3337, 3338, 3367, 3368, 3377, 3387, 3390, 3421, 3429, 3430, 3438, 3439, 3440, 3447, 3448, 3449, 3450,
           3454, 3455, 3456, 3485, 3486, 3487, 3496, 3535, 3538, 3539, 3542, 3543, 3546, 3548, 3576, 3587, 3619, 3622,
           3624, 3714, 3748, 4409, 4429, 4430, 4465, 4466, 4467, 4468, 4475, 4476, 4477, 4513, 4565, 4566, 4567, 4568,
           4569, 4570, 4594, 4595, 4596, 4597, 4698, 4714, 4715, 4716, 4717, 4723, 4725, 4732, 4738, 4817, 4891, 4998,
           5030, 5031, 5032, 5065, 5066, 5107, 5119, 5120, 5121, 5122, 5123, 5153, 5154, 5234, 5235, 5299, 5381, 5391,
           5392, 5393, 5394, 5395, 5396, 5433, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5498, 5510, 5965, 6300, 6991,
           7169, 7358, 8857, 9000, 9001, 9002, 9013, 9056, 9109, 9110, 9117, 9128, 9150, 9151, 9152, 9156, 9157, 9162,
           9171, 9177, 9183, 9184, 9185, 9186, 9187, 9195, 9196, 9197, 9234, 9235, 9236, 9237, 9238, 9239, 9272, 9273,
           9274, 9275, 9284, 9285, 9286, 9287, 9288, 9307, 9308, 9314, 9315, 9316, 9344, 9345, 9346, 9347, 9348, 9349,
           9369, 9370, 9371, 9372, 9461, 9462, 9539, 9540, 9712, 9713, 9714, 9715, 9716, 9717, 9740, 9741, 9742, 9758,
           9759, 9760, 9776, 9777, 9778, 9779, 9838, 9839, 9840, 9841, 9842, 9848, 9864, 9880, 9881, 9882, 9883, 9884,
           9885, 9945, 9946, 9975, 9994, 9995, 10029, 10030, 10031, 10032, 10065, 10067, 10069, 10110, 10126, 10127,
           10183, 10184, 10185, 10222, 10226, 10228, 208, 215, 997, 1037, 1100, 2279, 3366, 4280, 5155, 10058]



class FLIR(data.Dataset):
  num_classes = 3
  default_resolution = [384, 384]
  # mean = np.array([0.40789654, 0.44719302, 0.47026115],
  #                  dtype=np.float32).reshape(1, 1, 3)
  # std = np.array([0.28863828, 0.27408164, 0.27809835],
  #                  dtype=np.float32).reshape(1, 1, 3)

  mean = np.array([0.44719302, 0.44719302, 0.44719302],
                   dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.27809835, 0.27809835, 0.27809835],
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
    self.images_all = self.coco.getImgIds()
    self.images = []
    for img_i in self.images_all:
      if self.split == 'train':
        if not img_i+1 in exclude:
          self.images = self.images + [img_i]
      if self.split == 'val':
        if not img_i+8863 in exclude:
          self.images = self.images + [img_i]
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
