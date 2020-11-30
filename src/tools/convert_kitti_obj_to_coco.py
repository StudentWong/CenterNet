from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
exit()
import pickle
import random
import json
import numpy as np
import cv2
import os
import shutil
OUT_DATA_PATH = '../../data/kitti/'
IN_img_PATH = '/home/htz/caijihuzhuo/datasets/img/training/image_2'
IN_label_PATH = '/home/htz/caijihuzhuo/datasets/label/training/label_2'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import _init_paths


'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

cats = ['Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck',  'Person_sitting',
        'Tram', 'Misc', 'DontCare']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384 # 375
W = 1248 # 1242
EXT = [45.75, -0.34, 0.005]

cat_mapping = {'Pedestrian': 1, 'Car': 3, 'Cyclist': 2, 'Van': 3}
# cat_info = []

cat_info = [{'name': 'person', 'id': 1}, {'name': 'bicycle', 'id': 2}, {'name': 'car', 'id': 3}]

train_img_out_path = OUT_DATA_PATH + 'train/images'
val_img_out_path = OUT_DATA_PATH + 'val/images'

# original_image_set_path = DATA_PATH + 'ImageSets_{}/'.format(SPLIT)
# ann_dir = DATA_PATH + 'training/label_2/'

splits = ['train', 'val']
train_val_split_ratio = 0.133
imglist = os.listdir(IN_img_PATH)
labellist = os.listdir(IN_label_PATH)
assert len(labellist) == len(imglist)
data_len = len(labellist)
val_len = int(data_len*train_val_split_ratio)
train_len = data_len - val_len
data_len = {'train': train_len, 'val': val_len}
# print(train_len)
# exit()

for split in ['train', 'val']:
  ret = {'images': [], 'annotations': [], "categories": cat_info}
  while True:
      # image_set = open(image_set_path + '{}.txt'.format(split), 'r')
      # image_to_id = {}
      if len(ret['images']) >= data_len[split]:
          break
      img_name = random.choice(imglist)
      imglist.remove(img_name)
      idx = img_name.replace(".png", '')
      idx = int(idx)
      relative_img_path = 'images/' + img_name
      image_info = {'file_name': relative_img_path,
                    'id': idx+1}
      ret['images'].append(image_info)

      shutil.copyfile(os.path.join(IN_img_PATH, img_name),
                      os.path.join(OUT_DATA_PATH, split, relative_img_path))

      ann_path_original = os.path.join(IN_label_PATH, '{:06d}.txt'.format(idx))
      with open(ann_path_original, 'r') as annfile:
          for ann_ind, txt in enumerate(annfile):
              tmp = txt[:-1].split(' ')
              if tmp[0] not in cat_mapping.keys():
                  continue
              cat_id = cat_mapping[tmp[0]]
              truncated = int(float(tmp[1]))
              occluded = int(tmp[2])
              if occluded == 2:
                  continue
              alpha = float(tmp[3])
              bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
              dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
              location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
              rotation_y = float(tmp[14])

              ann = {'image_id': idx+1,
                     'id': int(len(ret['annotations']) + 1),
                     'category_id': cat_id,
                     'bbox': _bbox_to_coco_bbox(bbox)
                     }
              ret['annotations'].append(ann)
  out_label_path1 = os.path.join(OUT_DATA_PATH, 'annotations',
                                'kitti_{}.json'.format(split))
  out_label_path2 = os.path.join(OUT_DATA_PATH, split,
                                'kitti_{}.json'.format(split))
  # out_path = '{}/annotations/kitti_{}.json'.format(DATA_PATH, SPLIT, split)
  json.dump(ret, open(out_label_path1, 'w'), indent=4)
  json.dump(ret, open(out_label_path2, 'w'), indent=4)
  # print(len(ret['images']))
  # print(train_len)
  # exit()

  # for line in image_set:
  #   if line[-1] == '\n':
  #     line = line[:-1]
  #   image_id = int(line)
  #   image_info = {'file_name': '{}.png'.format(line),
  #                 'id': int(image_id),
  #                 'calib': calib.tolist()}
  #   ret['images'].append(image_info)
  #   if split == 'test':
  #     continue
  #   ann_path = ann_dir + '{}.txt'.format(line)
  #   # if split == 'val':
  #   #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
  #   anns = open(ann_path, 'r')
  #
  #   if DEBUG:
  #     image = cv2.imread(
  #       DATA_PATH + 'images/trainval/' + image_info['file_name'])
  #
  #   for ann_ind, txt in enumerate(anns):
  #     tmp = txt[:-1].split(' ')
  #     cat_id = cat_ids[tmp[0]]
  #     truncated = int(float(tmp[1]))
  #     occluded = int(tmp[2])
  #     alpha = float(tmp[3])
  #     bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
  #     dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
  #     location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
  #     rotation_y = float(tmp[14])
  #
  #     ann = {'image_id': image_id,
  #            'id': int(len(ret['annotations']) + 1),
  #            'category_id': cat_id,
  #            'dim': dim,
  #            'bbox': _bbox_to_coco_bbox(bbox),
  #            'depth': location[2],
  #            'alpha': alpha,
  #            'truncated': truncated,
  #            'occluded': occluded,
  #            'location': location,
  #            'rotation_y': rotation_y}
  #     ret['annotations'].append(ann)
  #     if DEBUG and tmp[0] != 'DontCare':
  #       box_3d = compute_box_3d(dim, location, rotation_y)
  #       box_2d = project_to_image(box_3d, calib)
  #       # print('box_2d', box_2d)
  #       image = draw_box_3d(image, box_2d)
  #       x = (bbox[0] + bbox[2]) / 2
  #       '''
  #       print('rot_y, alpha2rot_y, dlt', tmp[0],
  #             rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
  #             np.cos(
  #               rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
  #       '''
  #       depth = np.array([location[2]], dtype=np.float32)
  #       pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
  #                         dtype=np.float32)
  #       pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
  #       pt_3d[1] += dim[0] / 2
  #       print('pt_3d', pt_3d)
  #       print('location', location)
  #   if DEBUG:
  #     cv2.imshow('image', image)
  #     cv2.waitKey()


  # print("# images: ", len(ret['images']))
  # print("# annotations: ", len(ret['annotations']))
  # # import pdb; pdb.set_trace()
  # out_path = '{}/annotations/kitti_{}_{}.json'.format(DATA_PATH, SPLIT, split)
  # json.dump(ret, open(out_path, 'w'))

