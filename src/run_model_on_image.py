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
from logger import Logger
from datasets.dataset_factory import get_dataset, dataset_factory
from trains.train_factory import train_factory
from detectors.detector_factory import detector_factory
from utils.utils import AverageMeter
from progress.bar import Bar

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
        img_path_T = os.path.join(self.img_dir, img_info['file_name'])
        image_T = cv2.imread(img_path_T)

        img_path_R = img_path_T.replace("thermal_8_bit", "rgb_adjusted")
        img_path_R = img_path_R.replace("jpeg", "jpg")
        image_R = cv2.imread(img_path_R)

        image = [image_R, image_T]
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

if __name__ == "__main__":

    opt = opts().parse()
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv, opt.adapt_thermal_weight, opt.share_cut)
    load_model_fun = load_model
    model = load_model_fun(
        model, opt.load_model)
    # ckpt = torch.load(opt.load_model)
    # model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda()
    model.set_temp(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    # print(opt.mean)
    # print(opt.std)
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
    dataset.run_eval_return(results, opt.save_dir)
    # print(model)


