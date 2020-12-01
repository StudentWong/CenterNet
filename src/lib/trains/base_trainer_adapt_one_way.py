from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import itertools
from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter


class ModelWithLoss(torch.nn.Module):
  def __init__(self, model_detect, loss_detect, model_gan):
    super(ModelWithLoss, self).__init__()
    self.model_detect = model_detect
    self.model_gan = model_gan
    self.loss_detect = loss_detect

  def cal_sem_loss(self, batch):

    assert self.ft_Fake.shape == self.ft_True.shape
    assert batch['kitti_hm'].shape == batch['flir_hm'].shape
    assert self.ft_Fake.shape[0] == batch['kitti_hm'].shape[0]
    N = self.ft_Fake.shape[0]
    assert self.ft_Fake.shape[2] == batch['kitti_hm'].shape[2]
    W = self.ft_Fake.shape[2]
    assert self.ft_Fake.shape[3] == batch['kitti_hm'].shape[3]
    H = self.ft_Fake.shape[3]

    D = self.ft_Fake.shape[1]
    C = batch['kitti_hm'].shape[1]

    # N*D*W*H
    ft_fake_N_D_WH = self.ft_Fake.view(N, D, W*H)
    ft_true_N_D_WH = self.ft_True.view(N, D, W*H)

    # N*C*W*H
    hm_fake_N_WH_C = batch['kitti_hm'].view(N, C, W*H).permute(0, 2, 1).contiguous()
    hm_true_N_WH_C = batch['flir_hm'].view(N, C, W*H).permute(0, 2, 1).contiguous()

    ft_fake_N_D_C = torch.bmm(ft_fake_N_D_WH, hm_fake_N_WH_C)/(W*H)
    ft_true_N_D_C = torch.bmm(ft_true_N_D_WH, hm_true_N_WH_C)/(W*H)
    ft_fake_N_C = ft_fake_N_D_C.sum(dim=1)
    ft_true_N_C = ft_true_N_D_C.sum(dim=1)
    return ((ft_fake_N_C-ft_true_N_C)**2)*batch['kitti_have_object']*batch['flir_have_object']
    # print(ft_fake_N_D_C.shape)
    # print(batch['flir_have_object'].shape)
    # exit()
    # loss_sement =


  def forward(self, batch):
    # print(batch['flir_input'].shape)
    # print(batch.keys())
    # exit()
    self.model_gan.set_input(batch)
    self.model_gan.forward()
    # print(self.model_gan.fake_B.shape)
    # exit()
    # print(batch['flir_input'].shape)
    # print(self.model_gan.fake_B.shape)
    # outputs = self.model_detect(torch.cat([batch['flir_input'], self.model_gan.fake_B], dim=0))
    self.outputs_Fake = self.model_detect(self.model_gan.fake_B)
    self.outputs_True = self.model_detect(batch['flir_input'])

    # print(len(self.outputs_Fake))
    self.ft_Fake = self.outputs_Fake[-1]['ft_backbone']
    self.ft_True = self.outputs_True[-1]['ft_backbone']
    # N*D*W*H

    self.loss_sement = self.cal_sem_loss(batch).mean()
    # print(batch['flir_hm'].shape)
    # print(batch['kitti_hm'].shape)
    # print(outputs)
    # exit()
    # loss, loss_stats = self.loss(outputs, batch)
    # return outputs[-1], loss, loss_stats

  def backward_model_G(self, batch, optimizer_model_G, lossstat):

    self.model_gan.set_requires_grad(self.model_gan.netD_A, False)
    optimizer_model_G.zero_grad()

    lambda_idt = self.model_gan.opt.lambda_identity
    lambda_B = self.model_gan.opt.lambda_B
    # Identity loss
    if lambda_idt > 0:
      # G_A should be identity if real_B is fed.
      self.model_gan.idt_A = self.model_gan.netG_A(self.model_gan.real_B)
      self.model_gan.loss_idt_A = self.model_gan.criterionIdt(self.model_gan.idt_A, self.model_gan.real_B) * lambda_B * lambda_idt

    else:
      self.model_gan.loss_idt_A = 0

    # GAN loss D_A(G_A(A))
    self.model_gan.loss_G_A = self.model_gan.criterionGAN(self.model_gan.netD_A(self.model_gan.fake_B), True)
    loss_fake, loss_stats_fake = self.loss_detect(self.outputs_Fake, batch, "Fake")
    loss_true, loss_stats_true = self.loss_detect(self.outputs_True, batch, "True")


    # combined loss
    self.loss_model_G = self.model_gan.loss_G_A + \
                        self.model_gan.loss_idt_A + \
                        loss_fake + loss_true + \
                        self.loss_sement*200

    lossstat['loss_G_A'] = self.model_gan.loss_G_A
    lossstat['loss_idt_A'] = self.model_gan.loss_idt_A

    lossstat['flir_hm_loss'] = loss_stats_true['hm_loss']
    lossstat['flir_wh_loss'] = loss_stats_true['wh_loss']
    lossstat['flir_off_loss'] = loss_stats_true['off_loss']
    lossstat['kitti_hm_loss'] = loss_stats_fake['hm_loss']
    lossstat['kitti_wh_loss'] = loss_stats_fake['wh_loss']
    lossstat['kitti_off_loss'] = loss_stats_fake['off_loss']
    lossstat['loss_G'] = self.loss_model_G
    lossstat['sement']= self.loss_sement

    self.loss_model_G.backward()
    optimizer_model_G.step()
    # optimizer_model_G.zero_grad()

    del self.outputs_Fake, self.outputs_True, self.loss_model_G

  def backward_D(self, optimizer_D, lossstat):
    self.model_gan.set_requires_grad([self.model_gan.netD_A, self.model_gan.netD_B], True)
    optimizer_D.zero_grad()
    self.model_gan.backward_D_A()
    self.model_gan.backward_D_B()
    lossstat['loss_D_A'] = self.model_gan.loss_D_A
    lossstat['loss_D_B'] = self.model_gan.loss_D_B
    optimizer_D.step()

  def forward_T(self, batch):
    outputs = self.model.forward_T(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class BaseTrainer(object):
  def __init__(
    self, opt, model, gan):
    # print("asd")
    # exit()
    self.opt = opt
    # self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss, gan)

    params_model_G = [
      {"params": model.parameters(), "lr": opt.lr},
      {"params": gan.netG_A.parameters(), "lr": 2.5e-4, "betas": (opt.beta1, 0.999)},
    ]
    # optimizer = torch.optim.SGD(params, momentum=config.momentum, weight_decay=config.weight_decay)

    self.optimizer_model_G = torch.optim.Adam(params_model_G)
    self.optimizer_D = torch.optim.Adam(gan.netD_A.parameters(),
                                        lr=2.5e-4, betas=(opt.beta1, 0.999))

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer_model_G.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)
    for state in self.optimizer_D.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

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

      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True)
      # print(batch.keys())
      model_with_loss(batch)
      loss_stats = {'loss': 0, 'flir_hm_loss': 0, 'flir_wh_loss': 0, 'flir_off_loss': 0,
                   'kitti_hm_loss': 0, 'kitti_wh_loss': 0, 'kitti_off_loss': 0,
                   'loss_G_A': 0, 'loss_G_B': 0, 'loss_cycle_A': 0, 'loss_cycle_B': 0,
                   'loss_idt_A': 0, 'loss_idt_B': 0, 'sement': 0}
      model_with_loss.backward_model_G(batch, self.optimizer_model_G, loss_stats)
      model_with_loss.backward_D(self.optimizer_D, loss_stats)
      # print(loss_stats)
      # exit()
      # output, loss, loss_stats = model_with_loss(batch)
      # loss = loss.mean()
      # if phase == 'train':
      #   self.optimizer.zero_grad()
      #   loss.backward()
      #   self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['flir_input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      # if opt.debug > 0:
      #   self.debug(batch, output, iter_id)
      #
      # if opt.test:
      #   self.save_result(output, batch, results)
      del loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def debug(self, batch, output, iter_id):
    raise NotImplementedError

  def save_result(self, output, batch, results):
    raise NotImplementedError

  def _get_losses(self, opt):
    raise NotImplementedError
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)