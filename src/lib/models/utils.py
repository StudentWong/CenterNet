from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
# import torch.autograd.Function

zero_clip = 1e-6


def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _tanh(x):
  y = torch.clamp(x.tanh_(), min=-1+1e-4, max=1-1e-4)
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def sign(x, c):
    flag_gt = (x > c).float()
    flag_st = (x < c).float()
    return flag_gt-flag_st


class Custom_Activation_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, c, lamda):
        #input shape: N*C*(w*h)
        #c shape: C
        #lamda shape : C
        # print(input.shape)
        # print(c.shape)
        # print(lamda.shape)

        c_expand = c.unsqueeze(0).unsqueeze(2).expand_as(input).detach()
        lamda_expand = lamda.unsqueeze(0).unsqueeze(2).expand_as(input).detach()

        region1_flag = (torch.abs(input - c_expand) >= lamda_expand/2) * (torch.abs(input - c_expand) < lamda_expand)
        region2_flag = (torch.abs(input - c_expand) >= 0.0) * (torch.abs(input - c_expand) < lamda_expand/2)
        # print(region1_flag)
        # print(region2_flag)
        # print(region1_flag.float())
        # print(region2_flag.float())
        region1_flag = region1_flag.float().detach().requires_grad_(False)
        region2_flag = region2_flag.float().detach().requires_grad_(False)
        # print(region1_flag.shape)

        output_region1 = region1_flag * 2 * ((1-(torch.abs(input - c_expand)/lamda_expand))**2)
        output_region2 = region2_flag * (1 - 2*((torch.abs(input - c_expand)/lamda_expand)**2))

        output_clip1_pos = output_region1.clamp(min=zero_clip)
        output_clip1_neg = output_region1.clamp(max=-zero_clip)
        output_clip1 = output_clip1_pos + output_clip1_neg

        output_clip2_pos = output_region2.clamp(min=zero_clip)
        output_clip2_neg = output_region2.clamp(max=-zero_clip)
        output_clip2 = output_clip2_pos + output_clip2_neg
        # output = region1_flag * 2 * ((1-(torch.abs(input - c_expand)/lamda))**2) \
        #          + region2_flag * (1 - 2*((torch.abs(input - c_expand)/lamda)**2))
        output = output_clip1 + output_clip2
        # print(output.shape)
        ctx.save_for_backward(input, c_expand, lamda_expand, region1_flag, region2_flag)
        # ctx.save_for_backward(input, c, lamda)
        # print(ctx.saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output为反向传播上一级计算得到的梯度值
        # print(ctx.saved_tensors)
        input, c_expand, lamda_expand, region1_flag, region2_flag = ctx.saved_variables
        grad_input = grad_c = grad_lamda = None

        # print(ctx.needs_input_grad)
        if ctx.needs_input_grad[0]:
            grad_input1 = -((4 * (lamda_expand - torch.abs(input - c_expand))) / (lamda_expand ** 2)) * sign(input,
                                                                                                        c_expand) * region1_flag
            grad_input2 = -((4 * (input - c_expand)) / (lamda_expand ** 2)) * region2_flag

            grad_input1_pos = grad_input1.clamp(min=zero_clip)
            grad_input1_neg = grad_input1.clamp(max=-zero_clip)
            grad_input1_clip = grad_input1_pos + grad_input1_neg

            grad_input2_pos = grad_input2.clamp(min=zero_clip)
            grad_input2_neg = grad_input2.clamp(max=-zero_clip)
            grad_input2_clip = grad_input2_pos + grad_input2_neg

            grad_input = grad_input1_clip + grad_input2_clip
            # print(grad_input.shape)

        if ctx.needs_input_grad[1]:
            grad_c1 = ((4 * (lamda_expand - torch.abs(input - c_expand)))/(lamda_expand**2))*sign(input, c_expand)*region1_flag
            grad_c2 = ((4*(input - c_expand))/(lamda_expand**2))*region2_flag

            grad_c1_pos = grad_c1.clamp(min=zero_clip)
            grad_c1_neg = grad_c1.clamp(max=-zero_clip)
            grad_c1_clip = grad_c1_pos + grad_c1_neg

            grad_c2_pos = grad_c2.clamp(min=zero_clip)
            grad_c2_neg = grad_c2.clamp(max=-zero_clip)
            grad_c2_clip = grad_c2_pos + grad_c2_neg

            grad_c_expand = grad_c1_clip + grad_c2_clip
            grad_c = (grad_output * grad_c_expand).sum(dim=[0, 2]).detach()
            # print(grad_c_expand)
        if ctx.needs_input_grad[2]:
            grad_lamda1 = ((4*(lamda_expand - torch.abs(input - c_expand))*(input - c_expand))/(lamda_expand**3))*sign(input, c_expand)*region1_flag
            grad_lamda2 = ((4*((input - c_expand)**2))/(lamda_expand**3))*region2_flag
            grad_lamda1_pos = grad_lamda1.clamp(min=zero_clip)
            grad_lamda1_neg = grad_lamda1.clamp(max=-zero_clip)
            grad_lamda1_clip = grad_lamda1_pos + grad_lamda1_neg

            grad_lamda2_pos = grad_lamda2.clamp(min=zero_clip)
            grad_lamda2_neg = grad_lamda2.clamp(max=-zero_clip)
            grad_lamda2_clip = grad_lamda2_pos + grad_lamda2_neg

            grad_lamda_expand = grad_lamda1_clip + grad_lamda2_clip
            # print(grad_lamda_expand)
            # print((grad_output * grad_lamda_expand).shape)
            grad_lamda = (grad_output * grad_lamda_expand).sum(dim=[0, 2]).detach()
        # exit()
        # print(grad_input.shape)
        # print(grad_c.shape)
        # print(grad_lamda.shape)
        return grad_input, grad_c, grad_lamda

class Custom_Activation(nn.Module):
    def __init__(self, class_num):
        super(Custom_Activation, self).__init__()
        self.c = nn.Parameter(data=torch.randn(class_num))
        self.lamda = nn.Parameter(data=torch.normal(1*torch.ones(class_num)))

    def forward(self, input):
        return Custom_Activation_Function.apply(input, self.c, self.lamda)



class Custom_Activation_Function_Multi_dim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, c, lamda):
        #input shape: N*D*(w*h)
        #c shape: D*C
        #lamda shape : D*C
        #output: N*C*(w*h)

        #hidden: N*D*C*(w*h)

        # print(input.shape)
        # print(c.shape)
        # exit()
        N = input.shape[0]
        assert input.shape[1]==c.shape[0] and c.shape[0]==lamda.shape[0]
        D = input.shape[1]
        assert c.shape[1] == lamda.shape[1]
        C = c.shape[1]
        assert len(input.shape) == 3
        wh = input.shape[2]
        input_expand = input.unsqueeze(2).expand([N, D, C, wh]).detach()

        c_expand = c.unsqueeze(0).unsqueeze(3).expand_as(input_expand).detach()
        lamda_expand = lamda.unsqueeze(0).unsqueeze(3).expand_as(input_expand).detach()

        # print(c_expand.shape)
        # print(lamda_expand.shape)
        # exit()
        region1_flag = (torch.abs(input_expand - c_expand) >= lamda_expand/2) * (torch.abs(input_expand - c_expand) < lamda_expand)
        region2_flag = (torch.abs(input_expand - c_expand) >= 0.0) * (torch.abs(input_expand - c_expand) < lamda_expand/2)
        # print(region1_flag)
        # print(region2_flag)
        # print(region1_flag.float())
        # print(region2_flag.float())
        region1_flag = region1_flag.float().detach().requires_grad_(False)
        region2_flag = region2_flag.float().detach().requires_grad_(False)
        # print(region1_flag.shape)

        output_region1 = region1_flag * 2 * ((1-(torch.abs(input_expand - c_expand)/lamda_expand))**2)
        output_region2 = region2_flag * (1 - 2*((torch.abs(input_expand - c_expand)/lamda_expand)**2))

        output_clip1_pos = output_region1.clamp(min=zero_clip)
        output_clip1_neg = output_region1.clamp(max=-zero_clip)
        output_clip1 = output_clip1_pos + output_clip1_neg

        output_clip2_pos = output_region2.clamp(min=zero_clip)
        output_clip2_neg = output_region2.clamp(max=-zero_clip)
        output_clip2 = output_clip2_pos + output_clip2_neg
        # output = region1_flag * 2 * ((1-(torch.abs(input - c_expand)/lamda))**2) \
        #          + region2_flag * (1 - 2*((torch.abs(input - c_expand)/lamda)**2))
        # print((output_clip1 + output_clip2).shape)
        output = (output_clip1 + output_clip2).prod(dim=1)
        # print(output.shape)
        # exit()
        ctx.save_for_backward(input_expand, c_expand, lamda_expand, region1_flag, region2_flag)
        # ctx.save_for_backward(input, c, lamda)
        # print(ctx.saved_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # input shape: N*D*(w*h)
        # c shape: D*C
        # lamda shape : D*C
        # output: N*C*(w*h)

        # hidden: N*D*C*(w*h)

        input_expand, c_expand, lamda_expand, region1_flag, region2_flag = ctx.saved_variables
        grad_input = grad_c = grad_lamda = None
        # print(input_expand.shape)
        # print(grad_output.shape)
        grad_output_expand = grad_output.unsqueeze(1).expand_as(input_expand)
        # print(input_expand.shape)
        # print(ctx.needs_input_grad)
        if ctx.needs_input_grad[0]:
            grad_input1 = -((4 * (lamda_expand - torch.abs(input_expand - c_expand))) / (lamda_expand ** 2)) * sign(input_expand,
                                                                                                        c_expand) * region1_flag
            grad_input2 = -((4 * (input_expand - c_expand)) / (lamda_expand ** 2)) * region2_flag

            grad_input1_pos = grad_input1.clamp(min=zero_clip)
            grad_input1_neg = grad_input1.clamp(max=-zero_clip)
            grad_input1_clip = grad_input1_pos + grad_input1_neg

            grad_input2_pos = grad_input2.clamp(min=zero_clip)
            grad_input2_neg = grad_input2.clamp(max=-zero_clip)
            grad_input2_clip = grad_input2_pos + grad_input2_neg

            grad_input = (grad_input1_clip + grad_input2_clip).sum(dim=2)
            # print(grad_input.shape)
            # exit()

        if ctx.needs_input_grad[1]:
            grad_c1 = ((4 * (lamda_expand - torch.abs(input_expand - c_expand)))/(lamda_expand**2))*sign(input_expand, c_expand)*region1_flag
            grad_c2 = ((4*(input_expand - c_expand))/(lamda_expand**2))*region2_flag

            grad_c1_pos = grad_c1.clamp(min=zero_clip)
            grad_c1_neg = grad_c1.clamp(max=-zero_clip)
            grad_c1_clip = grad_c1_pos + grad_c1_neg

            grad_c2_pos = grad_c2.clamp(min=zero_clip)
            grad_c2_neg = grad_c2.clamp(max=-zero_clip)
            grad_c2_clip = grad_c2_pos + grad_c2_neg

            grad_c_expand = grad_c1_clip + grad_c2_clip
            # print(grad_output.shape)
            # print(grad_c_expand.shape)
            grad_c = (grad_output_expand * grad_c_expand).sum(dim=[0, 3]).detach()
            # print(grad_c_expand)
            # print()
        if ctx.needs_input_grad[2]:
            grad_lamda1 = ((4*(lamda_expand - torch.abs(input_expand - c_expand))*(input_expand - c_expand))/(lamda_expand**3))*sign(input_expand, c_expand)*region1_flag
            grad_lamda2 = ((4*((input_expand - c_expand)**2))/(lamda_expand**3))*region2_flag
            grad_lamda1_pos = grad_lamda1.clamp(min=zero_clip)
            grad_lamda1_neg = grad_lamda1.clamp(max=-zero_clip)
            grad_lamda1_clip = grad_lamda1_pos + grad_lamda1_neg

            grad_lamda2_pos = grad_lamda2.clamp(min=zero_clip)
            grad_lamda2_neg = grad_lamda2.clamp(max=-zero_clip)
            grad_lamda2_clip = grad_lamda2_pos + grad_lamda2_neg

            grad_lamda_expand = grad_lamda1_clip + grad_lamda2_clip
            # print(grad_lamda_expand)
            # print((grad_output * grad_lamda_expand).shape)
            grad_lamda = (grad_output_expand * grad_lamda_expand).sum(dim=[0, 3]).detach()
        # exit()
        # print(grad_input.shape)
        # print(grad_c.shape)
        # print(grad_lamda.shape)
        return grad_input, grad_c, grad_lamda



class Custom_Activation_Multi_dim(nn.Module):
    def __init__(self, feature, class_num):
        super(Custom_Activation_Multi_dim, self).__init__()
        self.c = nn.Parameter(data=torch.randn(feature, class_num))
        self.lamda = nn.Parameter(data=torch.normal(1*torch.ones(feature, class_num)))

    def forward(self, input):
        return Custom_Activation_Function_Multi_dim.apply(input, self.c, self.lamda)


class Custom_Activation_Multi_dim_torch_back(nn.Module):
    def __init__(self, feature, class_num):
        super(Custom_Activation_Multi_dim_torch_back, self).__init__()
        # self.c = nn.Parameter(data=torch.randn(feature, class_num, dtype=torch.float))
        # self.c = nn.Parameter(data=torch.zeros(feature, class_num, dtype=torch.float)))
        self.c = nn.Parameter(data=0.0*torch.ones((feature, class_num), dtype=torch.float))
        # self.lamda = nn.Parameter(data=torch.normal(1*torch.ones(feature, class_num, dtype=torch.float)))
        # self.lamda = nn.Parameter(0.5 * torch.ones((feature, class_num), dtype=torch.float), requires_grad=False)

        self.lamda = nn.Parameter(2 * torch.ones((feature, class_num), dtype=torch.float))

    def forward(self, input):
        N = input.shape[0]
        # print(input.shape[1])
        # print(self.c.shape[1])
        assert input.shape[1] == self.c.shape[0] and self.c.shape[0] == self.lamda.shape[0]
        D = input.shape[1]
        assert self.c.shape[1] == self.lamda.shape[1]
        C = self.c.shape[1]
        assert len(input.shape) == 3
        wh = input.shape[2]
        input_expand = input.unsqueeze(2).expand([N, D, C, wh])

        c_expand = self.c.unsqueeze(0).unsqueeze(3).expand_as(input_expand)
        lamda_expand = self.lamda.unsqueeze(0).unsqueeze(3).expand_as(input_expand)
        lamda_valid_flag = lamda_expand>0.0
        lamda_valid_flag = lamda_valid_flag.float().detach().requires_grad_(False)

        region1_flag = (torch.abs(input_expand - c_expand) >= lamda_expand / 2) * (
                    torch.abs(input_expand - c_expand) < lamda_expand)
        region2_flag = (torch.abs(input_expand - c_expand) >= 0.0) * (
                    torch.abs(input_expand - c_expand) < lamda_expand / 2)

        region1_flag = region1_flag.float().detach().requires_grad_(False)
        region2_flag = region2_flag.float().detach().requires_grad_(False)

        output_region1 = lamda_valid_flag * region1_flag * 2 * ((1 - (torch.abs(input_expand - c_expand) / lamda_expand)) ** 2)
        output_region2 = lamda_valid_flag * region2_flag * (1 - 2 * ((torch.abs(input_expand - c_expand) / lamda_expand) ** 2))

        output_clip1_pos = output_region1.clamp(min=zero_clip)
        output_clip1_neg = output_region1.clamp(max=-zero_clip)
        output_clip1 = output_clip1_pos + output_clip1_neg

        output_clip2_pos = output_region2.clamp(min=zero_clip)
        output_clip2_neg = output_region2.clamp(max=-zero_clip)
        output_clip2 = output_clip2_pos + output_clip2_neg
        output = (output_clip1 + output_clip2).prod(dim=1)
        return output


# import numpy as np
# if __name__ == '__main__':
#
#     x_in1 = np.linspace(-1, 1, num=100)
#     x_in2 = np.linspace(-1, 1, num=100)
#     x_in = np.zeros((10000, 2), dtype=np.float)
#     y = np.zeros((10000, 4), dtype=np.float)
#
#     for i, x1 in enumerate(x_in1):
#         for j, x2 in enumerate(x_in2):
#             x_in[i * 100 + j][0] = x1
#             x_in[i * 100 + j][1] = x2
#             if x1 > 0 and x2 > 0:
#                 y[i * 100 + j][0] = np.sqrt((-4 * (x1 - 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 - 0.5) ** 2 + 1).clip(min=1e-5))
#             elif x1 > 0 and x2 < 0:
#                 y[i * 100 + j][1] = np.sqrt((-4 * (x1 - 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 + 0.5) ** 2 + 1).clip(min=1e-5))
#             elif x1 < 0 and x2 < 0:
#                 y[i * 100 + j][2] = np.sqrt((-4 * (x1 + 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 + 0.5) ** 2 + 1).clip(min=1e-5))
#             elif x1 < 0 and x2 > 0:
#                 y[i * 100 + j][3] = np.sqrt((-4 * (x1 + 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 - 0.5) ** 2 + 1).clip(min=1e-5))
#
#     # x_in = x_in1
#     # y=(-4 * (x_in - 0.5) ** 2 + 1).clip(min=0.0)
#     # print(x_in)
#     # print(y)
#     # exit()
#
#     x_in_tensor = torch.tensor(x_in, requires_grad=False, dtype=torch.float).unsqueeze(2)
#     y_label_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float).unsqueeze(2)
#
#     layer = Custom_Activation_Multi_dim_torch_back(2, 4)
#     # x = torch.tensor([[[0.9, 0.1], [0.9, 0.1]], [[-0.9, 0.1], [-0.1, -2.5]]], dtype=torch.float, requires_grad=True)
#     # x2 = x ** 2
#     # print(x2.requires_grad)
#     # print(x.shape)
#     # print(layer(x))
#     # print(layer.c)
#     # print(x.shape)
#     loss_fun = torch.nn.MSELoss()
#     optim = torch.optim.SGD(layer.parameters(), lr=1)
#     for i in range(0, 10000):
#         y = layer(x_in_tensor)
#         # print(y.squeeze(2))
#
#         loss = loss_fun(y, y_label_tensor) \
#                # + 0.5*((0.0-layer.lamda).clamp(min=1e-5).sum()) \
#                # - 0.5 * ((0.6 - layer.lamda).clamp(max=-1e-5).sum()) \
#             # + 0.5*((layer.c-1).clamp(min=1e-5).sum())\
#             # - 0.5*((1+layer.c).clamp(max=-1e-5).sum()) #\
#
#         optim.zero_grad()
#         loss.backward()
#         print(layer.c.grad)
#         print(layer.lamda.grad)
#         optim.step()
#         print(layer.lamda)
#         print(layer.c)
#
#     exit()
#
#     Md = testmodel()
#     loss = torch.nn.L1Loss()
#
#     params = [
#         {"params": Md.layer1.parameters(), "lr": 1e-5},
#         # {"params": Md.net1.parameters(), "lr": 1e-4},
#     ]
#
#
#
#     # Md.layer1.c =
#     bestloss = 1e5
#     bestnetweight = []
#
#     for i in range(50000):
#         o = Md(x_in_tensor)
#         l = loss(o, y_label_tensor) \
#           + 0.5*((0.0-Md.layer1.lamda).clamp(min=1e-5).sum()) \
#           - 0.5 * ((1.2 - Md.layer1.lamda).clamp(max=-1e-5).sum()) \
#             + 0.5*((Md.layer1.c-1.5).clamp(min=1e-5).sum())\
#             - 0.5*((1.5+Md.layer1.c).clamp(max=-1e-5).sum()) #\
#
#
#         l.backward()
#         optim.step()
#
#         # Md.layer1.c = Md.layer1.c.clamp(min=-1, max=1)
#         # Md.layer1.lamda = Md.layer1.lamda.clamp(min=0.1, max=0.8)
#
#         if i % 50 == 0:
#             print('loss:')
#             print(l)
#             if l < bestloss:
#                 bestnetweight = Md.state_dict()
#             print('c:')
#             print(Md.layer1.c)
#             print('lamda:')
#             print(Md.layer1.lamda)
#
#
#     Md.load_state_dict(bestnetweight)
#     print('c:')
#     print(Md.layer1.c)
#     print('lamda:')
#     print(Md.layer1.lamda)
#     exit()
#     x_in_show = np.linspace(-1, 1, num=500)
#     x_in_show_tensor = torch.tensor(x_in_show, requires_grad=False, dtype=torch.float).unsqueeze(1)
#     # print(x_in_show_tensor)
#     y_label_show_tensor = Md(x_in_show_tensor)
#     # print(y_label_show_tensor)
#     import matplotlib.pyplot as plt
#
#     plt.plot(x_in_show_tensor.detach().numpy(), y_label_show_tensor.detach().numpy())
#     plt.show()



