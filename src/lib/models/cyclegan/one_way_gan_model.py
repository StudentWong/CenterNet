import torch
import itertools
from .image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class OneWayGANModel(BaseModel):
    def name(self):
        return 'OneWayGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizers = []
            # self.optimizers.append(self.optimizer_G)
            # self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = True
        self.real_A = input['kitti_input' if AtoB else 'flir_input'].to(self.device)
        self.real_B = input['flir_input' if AtoB else 'kitti_input'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        shape = self.fake_B.shape
        assert shape[1] == 3
        self.fake_B = self.fake_B.mean(dim=1, keepdim=True).expand(shape)


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G = self.loss_G_A + self.loss_idt_A
        self.loss_G.backward()

    # def optimize_parameters(self):
    #     # forward
    #     self.forward()
    #     # G_A and G_B
    #     self.set_requires_grad([self.netD_A, self.netD_B], False)
    #     self.optimizer_G.zero_grad()
    #     self.backward_G()
    #     self.optimizer_G.step()
    #     # D_A and D_B
    #     self.set_requires_grad([self.netD_A, self.netD_B], True)
    #     self.optimizer_D.zero_grad()
    #     self.backward_D_A()
    #     self.backward_D_B()
    #     self.optimizer_D.step()
