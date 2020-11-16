from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

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
import cv2


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
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
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    # print(model.state_dict().keys())
    # exit()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model_fun(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    # if opt.arch == 'dlaNoBias_34':
    #     optimizer = torch.optim.Adam(model.menber_activation.parameters(), opt.lr)

    Trainer = train_factory[opt.task]
    trainer = Trainer(opt, model, optimizer)
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
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model_fun(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                           epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)

            apar = prefetch_test(opt, model)
            # print(apar)

            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            for k, v in apar.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if apar['AP'] > ap_best:
                save_model_fun(os.path.join(opt.save_dir,
                                            'ap50_{:0.3f}_ap{:0.3f}_ar{:0.3f}.pth'.format(apar['AP50'], apar['AP'],
                                                                                          apar['AR100'])),
                               epoch, model)
                ap_best = apar['AP']
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model_fun(os.path.join(opt.save_dir, 'model_best.pth'),
                               epoch, model)
        else:
            save_model_fun(os.path.join(opt.save_dir, 'model_last.pth'),
                           epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model_fun(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                           epoch, model, optimizer)
            lr = opt.lr * (opt.lr_step_ratio ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
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
        return img_id, {'images': images, 'image': image, 'meta': meta, 'image_R': image_R, 'image_T': image_T}

    def __len__(self):
        return len(self.images)


def prefetch_test(opt, model):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = dataset_factory[opt.dataset]
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
        # print(pre_processed_images)
        results = ret['results']
        color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0]]
        cats = results.keys()
        for cat in cats:
            # print(cat)
            boxs = results[cat]
            # print(results[cat])
            # exit()
            for box in boxs:
                # print(box)
                if box[4]>0.2:
                    np_R = pre_processed_images['image_R'].detach().cpu().numpy()[0]
                    img_show_R = cv2.rectangle(np_R.astype(np.uint8), (box[0], box[1]), (box[2], box[3]), color[cat], 2)
                    cv2.imshow("show", img_show_R)
                    cv2.waitKey(0)

                    np_T = pre_processed_images['image_T'].detach().cpu().numpy()[0]
                    img_show_T = cv2.rectangle(np_T.astype(np.uint8), (box[0], box[1]), (box[2], box[3]), color[cat], 2)
                    cv2.imshow("show", img_show_T)
                    cv2.waitKey(0)
                    # exit()
        # print(ret)
        # exit()
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
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    # print(model.state_dict().keys())
    # exit()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
    model = model.cuda()
    prefetch_test(opt, model)
