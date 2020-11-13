from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import pandas as pd
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

# lamda='00'
data='val'
cat_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
color = ['aliceblue', 'black', 'brown', 'crimson', 'red',
         'darkgreen', 'darkkhaki', 'darkred', 'darkviolet', 'deeppink',
         'gold', 'greenyellow', 'ivory', 'lightblue', 'lightpink',
         'mediumblue', 'mediumorchid', 'olive', 'orange', 'pink']
color_background = 'gray'

fet_name = []
for i in range(64):
  fet_name = fet_name + ["feat" + str(i)]


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


  if data == "train":
    data_loader = train_loader
  elif data == "val":
    data_loader = val_loader

  num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
  bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)

  feature_result = []

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
      # mask = batch['hm'].eq(1).float().unsqueeze(1).expand((1, 64, 20, 96, 96)).detach()
      # map = (mask * outputs[-1]['ft'].unsqueeze(2).expand((1, 64, 20, 96, 96))).detach()
      mask_np = batch['hm'].detach().cpu().numpy()
      index = np.argwhere(mask_np==1)
      obj_num_pic = len(index)
      for num_obj in range(obj_num_pic):
        feature = outputs[-1]['ft'][0, :, index[num_obj][2], index[num_obj][3]].detach().cpu().numpy().tolist()
        # print(len(feature))
        # exit()
        feature = feature + [cat_name[index[num_obj][1]]]
        # print(feature)
        # exit()
        feature_result = feature_result + [feature]

      idx_background = np.random.randint(low=0, high=96, size=(obj_num_pic, 2))
      for num_obj in range(obj_num_pic):
        feature = outputs[-1]['ft'][0, :, idx_background[num_obj][0], idx_background[num_obj][1]].detach().cpu().numpy().tolist()
        feature = feature + ["background"]
        feature_result = feature_result + [feature]
      # print(feature_result)
      # exit()
      # print(mask.sum(dim=(0, 3, 4)))
      # print(map.sum(dim=(0, 3, 4)).detach())


      # print(outputs[-1]['ft'].max())

    # print(ft_sum)
    # print(mask_sum)
    # mean = ft_sum / mask_sum
    # print(mean)
    # print(model.menber_activation.c)
    # exit()
    print("pic:[{:d}/{:d}]".format(iter_id, num_iters))

  df = pd.DataFrame(feature_result, columns=fet_name + ["cat"])
  df.to_csv("./analysis/feature_cat_original.csv")
  # np.save("./analysis/std_np"+lamda+data, std_np)
  # np.save("./analysis/lamda_np"+lamda+data, lamda_np)
  # logger.close()

# if __name__ == '__main__':
#   opt = opts().parse()
#
#   test(opt)

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  test = "ori"
  if test == "ori":
    df = pd.read_csv("./analysis/feature_cat_original.csv")
    folder = 'pic_ori'
    range = (-1, 20)
  if test == "mem":
    df = pd.read_csv("./analysis/feature_cat.csv")
    folder = 'pic'
    range = (-1, 1)


  # range = None
  for i, cat in enumerate(cat_name):
    # print(df[df.cat == cat][fet_name + ["cat"]])
    # df[df.cat == cat][fet_name[0]]
    data = df[df.cat == cat][fet_name[0]].values
    # print(data)
    plt.figure(i)
    plt.hist(data, range=range, bins=160, facecolor=color[i], edgecolor="black")
    # plt.show()
    plt.savefig('./analysis/'+folder + '/' + cat + '.jpg')
    plt.close(i)


  data = df[df.cat == "background"][fet_name[0]].values
  plt.figure(20)
  plt.hist(data, range=range, bins=160, facecolor=color_background, edgecolor="black")
  # plt.show()
  plt.savefig('./analysis/'+folder + '/' + 'background' + '.jpg')

#
# if __name__ == "__main__":
#
#   data1 = [[1, 2, 3]]
#   data2 = [[4, 5, 6]]
#   df = pd.DataFrame(data1+data2,columns=["feature1", "feature2", "feature3"])
#   print(df)



# if __name__ == '__main__':
#   train05_std_np = np.load("./analysis/std_np05train.npy")
#   train05_lamda_np = np.load("./analysis/lamda_np05train.npy")
#   train05_mean_np = np.load("./analysis/mean_np05train.npy")
#   train05_c_np = np.load("./analysis/c_np05train.npy")
#
#   val05_std_np = np.load("./analysis/std_np05val.npy")
#   val05_lamda_np = np.load("./analysis/lamda_np05val.npy")
#   val05_mean_np = np.load("./analysis/mean_np05val.npy")
#   val05_c_np = np.load("./analysis/c_np05val.npy")
#
#   train01_std_np = np.load("./analysis/std_np01train.npy")
#   train01_lamda_np = np.load("./analysis/lamda_np01train.npy")
#   train01_mean_np = np.load("./analysis/mean_np01train.npy")
#   train01_c_np = np.load("./analysis/c_np01train.npy")
#
#   val01_std_np = np.load("./analysis/std_np01val.npy")
#   val01_lamda_np = np.load("./analysis/lamda_np01val.npy")
#   val01_mean_np = np.load("./analysis/mean_np01val.npy")
#   val01_c_np = np.load("./analysis/c_np01val.npy")
#
#   train002_std_np = np.load("./analysis/std_np002train.npy")
#   train002_lamda_np = np.load("./analysis/lamda_np002train.npy")
#   train002_mean_np = np.load("./analysis/mean_np002train.npy")
#   train002_c_np = np.load("./analysis/c_np002train.npy")
#
#   val002_std_np = np.load("./analysis/std_np002val.npy")
#   val002_lamda_np = np.load("./analysis/lamda_np002val.npy")
#   val002_mean_np = np.load("./analysis/mean_np002val.npy")
#   val002_c_np = np.load("./analysis/c_np002val.npy")
#
#   train00_std_np = np.load("./analysis/std_np00train.npy")
#   train00_lamda_np = np.load("./analysis/lamda_np00train.npy")
#   train00_mean_np = np.load("./analysis/mean_np00train.npy")
#   train00_c_np = np.load("./analysis/c_np00train.npy")
#
#   val00_std_np = np.load("./analysis/std_np00val.npy")
#   val00_lamda_np = np.load("./analysis/lamda_np00val.npy")
#   val00_mean_np = np.load("./analysis/mean_np00val.npy")
#   val00_c_np = np.load("./analysis/c_np00val.npy")
#
#
#   # for i in range(20):
#   #   print(np.corrcoef(train_mean_np[:, i], val_mean_np[:, i], rowvar=False)[0, 1])
#
#   # for i in range(20):
#   #   for j in range(20):
#   #     if i != j:
#   #       print(np.corrcoef(train_mean_np[:, i], val_mean_np[:, j], rowvar=False)[0, 1])
#   print("05:")
#   print(train05_std_np.sum(axis=0).mean())
#   print(val05_std_np.sum(axis=0).mean())
#   print((train05_std_np.sum(axis=0) / val05_std_np.sum(axis=0)).mean())
#
#   print("01:")
#   print(train01_std_np.sum(axis=0).mean())
#   print(val01_std_np.sum(axis=0).mean())
#   print((train01_std_np.sum(axis=0) / val01_std_np.sum(axis=0)).mean())
#
#   print("002:")
#   print(train002_std_np.sum(axis=0).mean())
#   print(val002_std_np.sum(axis=0).mean())
#   print((train002_std_np.sum(axis=0) / val002_std_np.sum(axis=0)).mean())
#
#   print("00:")
#   print(train00_std_np.sum(axis=0).mean())
#   print(val00_std_np.sum(axis=0).mean())
#   print((train00_std_np.sum(axis=0) / val00_std_np.sum(axis=0)).mean())
