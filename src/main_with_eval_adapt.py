from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import torch
import torch.utils.data
import numpy as np
from opts import opts
from models.model import create_model, load_model, save_model, load_model_freeze, save_model_freeze
from models.data_parallel import DataParallel
from models.cyclegan.cycle_gan_model import CycleGANModel
from logger import Logger
from datasets.dataset_factory import get_dataset, dataset_factory
from trains.train_factory import train_factory
from detectors.detector_factory import detector_factory
from utils.utils import AverageMeter
from progress.bar import Bar



def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = False
  Dataset = get_dataset(opt.dataset, opt.task)

  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if opt.task != 'ctdetfreeze':
    save_model_fun = save_model
    load_model_fun = load_model
  else:
    save_model_fun = save_model_freeze
    load_model_fun = load_model_freeze
  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt.adapt_thermal_weight, opt.share_cut)
  gan = CycleGANModel()
  gan.initialize(opt)
  # exit()
  # print(model)
  # exit()
  # print(model.state_dict().keys())
  # exit()
  # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  # if opt.load_model != '':
  #   model, optimizer, start_epoch = load_model_fun(
  #     model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  # if opt.arch == 'dlaNoBias_34':
  #     optimizer = torch.optim.Adam(model.menber_activation.parameters(), opt.lr)

  Trainer = train_factory[opt.task]
  print(Trainer)
  trainer = Trainer(opt, model, gan)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'),
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  ap_best = 0.0
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    # if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
    #   save_model_fun(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
    #              epoch, model)
    #   with torch.no_grad():
    #     log_dict_val, preds = trainer.val(epoch, val_loader)

    apar = prefetch_test(opt, model)
    # print(apar)


    # for k, v in log_dict_val.items():
    #   logger.scalar_summary('val_{}'.format(k), v, epoch)
    #   logger.write('{} {:8f} | '.format(k, v))
    for k, v in apar.items():
      logger.scalar_summary('val_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if apar['AP'] > ap_best:
        save_model_fun(os.path.join(opt.save_dir, 'ap50_{:0.3f}_ap{:0.3f}_ar{:0.3f}.pth'.format(apar['AP50'], apar['AP'], apar['AR100'])),
                        epoch, model)
        ap_best = apar['AP']
      # if log_dict_val[opt.metric] < best:
      #   best = log_dict_val[opt.metric]
      #   save_model_fun(os.path.join(opt.save_dir, 'model_best.pth'),
      #              epoch, model)
    else:
      save_model_fun(os.path.join(opt.save_dir, 'model_last.pth'),
                 epoch, model)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model_fun(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                 epoch, model)
      lr = opt.lr * (opt.lr_step_ratio ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in trainer.optimizer_model_G.param_groups:
          param_group['lr'] = lr
      for param_group in trainer.optimizer_D.param_groups:
          param_group['lr'] = lr
  logger.close()


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.opt = opt

    def __getitem__(self, index):
        img_id = self.images[index]
        img_info = self.load_image_func(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)

        images, meta = {}, {}
        for scale in opt.test_scales:
            if opt.task == 'ddd':
                images[scale], meta[scale] = self.pre_process_func(
                    image, scale, img_info['calib'])
            else:
                images[scale], meta[scale] = self.pre_process_func(image, scale)
        return img_id, {'images': images, 'image': image, 'meta': meta}

    def __len__(self):
        return len(self.images)

def prefetch_test(opt, model):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory["flir"]
    # Dataset_test = get_dataset("flir", "ctdet")
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Detector = detector_factory[opt.task]

    split = 'val' if not opt.trainval else 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt, True, model)
    # detector = Detector(opt, model=model)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind, (img_id, pre_processed_images) in enumerate(data_loader):
        # print(img_id)
        # exit()
        # time_begin = time()
        ret = detector.run(pre_processed_images)
        results[img_id.numpy().astype(np.int64)[0]] = ret['results']
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
                t, tm=avg_time_stats[t])
        bar.next()
    bar.finish()
    return dataset.run_eval_return(results, opt.save_dir)

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)