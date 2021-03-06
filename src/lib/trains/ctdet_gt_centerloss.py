from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import time
from progress.bar import Bar
from models.losses import FocalLoss, CenterLoss_gt
from models.networks.MemberShip_cuda.centerloss_cuda import CenterLoss_gt_cuda, CenterLoss_gt_eq1_cuda
from utils.utils import AverageMeter
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid, _tanh
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer

class CtdetLoss_NFS(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss_NFS, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    # self.centerloss = CenterLoss_gt()
    if opt.eq1:
      print("eq1 is True")
      self.centerloss = CenterLoss_gt_eq1_cuda()
    else:
      print("eq1 is False")
      self.centerloss = CenterLoss_gt_cuda()
    # self.crit = torch.nn.MSELoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt


  def forward(self, outputs, batch):
    opt = self.opt
    center_loss, hm_loss, wh_loss, off_loss = 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        pass
        # output['hm'] = _sigmoid(output['hm'])
        # print(_tanh(output['hm']))
        # print(output['hm'])
        # exit()
      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      # print(output['hm'])
      # print(batch['hm'])
      # print(self.crit(output['hm'], batch['hm']))
      # exit()
      # hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      ft_shape = output['ft'].shape
      hm_shape = output['hm'].shape
      # print(ft_shape)
      # print(hm_shape)
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      # print(hm_loss)
      # print(output['hm'])
      # print(batch['hm'])
      # print(self.crit(output['hm'], batch['hm']))
      # exit()

      if opt.center_weight > 0:
        center_loss += self.centerloss(output['ft'].view(ft_shape[0], ft_shape[1], -1),
                                    output['center'],
                                    output['hm'].view(hm_shape[0], hm_shape[1], -1),
                                       batch['hm']) / opt.num_stacks

      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss + opt.center_weight * center_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss, 'center_loss': center_loss}
    return loss, loss_stats



class CtdetTrainer_GT_Centerloss(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer_GT_Centerloss, self).__init__(opt, model, optimizer=optimizer)
    # print(self.optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss']
    if opt.wh_weight > 0:
      loss_states = loss_states + ['wh_loss']
    if opt.reg_offset and opt.off_weight > 0:
      loss_states = loss_states + ['off_loss']
    if opt.center_weight > 0:
      loss_states = loss_states + ['center_loss']

    loss = CtdetLoss_NFS(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      # print(self.model_with_loss.model.menber_activation.c)
      # print(self.model_with_loss.model.menber_activation.lamda)
      #['input', 'hm', 'reg_mask', 'ind', 'wh', 'reg', 'meta']
      # print(batch['input'].shape)
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      # print(batch)
      # exit()
      output, loss, loss_stats = model_with_loss(batch)
      # print(output)
      # exit()
      # print(loss)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                  '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
      else:
        bar.next()

      if opt.debug > 0:
        self.debug(batch, output, iter_id)

      if opt.test:
        self.save_result(output, batch, results)
      del output, loss, loss_stats

    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results