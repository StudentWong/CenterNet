import torch
import torch.nn
from torch import nn
from torch.utils.checkpoint import checkpoint


zero_clip = 1e-6
# c_leak = 0
# lamda_leak = 0

def norm_grad(input, max_norm):
    if input.requires_grad:
        def norm_hook(grad):
            N = grad.size(0) # batch number
            norm = grad.contiguous().view(N, -1).norm(p=2, dim=1) + 1e-6
            scale = (norm / max_norm).clamp(min=1).view([N]+[1]*(grad.dim()-1))
            return grad / scale

            # clip_coef = float(max_norm) / (grad.norm(2).data[0] + 1e-6)
            # return grad.mul(clip_coef) if clip_coef < 1 else grad
        input.register_hook(norm_hook)

class Membership_Activation(nn.Module):
    def __init__(self, feature, class_num, init_c=None, init_lamda=None, c_grad=True, lamda_grad=True):
        super(Membership_Activation, self).__init__()

        if init_c is None:
            self.c = nn.Parameter(data=0.0*torch.ones((feature, class_num), dtype=torch.float),
                                  requires_grad=c_grad)
        else:
            assert len(init_c.shape) == 2 and \
                   init_c.shape[0] == feature and \
                   init_c.shape[1] == class_num
            self.c = nn.Parameter(data=init_c, requires_grad=c_grad)

        if init_lamda is None:
            self.lamda = nn.Parameter(data=1.0*torch.ones((feature, class_num), dtype=torch.float),
                                      requires_grad=lamda_grad)
        else:
            assert len(init_lamda.shape) == 2 and \
                   init_lamda.shape[0] == feature and \
                   init_lamda.shape[1] == class_num
            self.lamda = nn.Parameter(data=init_lamda, requires_grad=lamda_grad)

    def forward(self, input):
        # input shape: N*D*(w*h)
        # c shape: D*C
        # lamda shape : D*C
        # output: N*C*(w*h)

        N = input.shape[0]
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

        # region_out_st = (input_expand < c_expand) * (~region1_flag * ~region2_flag)
        # region_out_gt = (input_expand > c_expand) * (~region1_flag * ~region2_flag)
        #
        #
        region1_flag = region1_flag.float().detach().requires_grad_(False)
        region2_flag = region2_flag.float().detach().requires_grad_(False)
        #
        # region_out_st = region_out_st.float().detach().requires_grad_(False)
        # region_out_gt = region_out_gt.float().detach().requires_grad_(False)
        # output_region_out_st = region_out_st*(c_leak*(input_expand-c_expand)-lamda_leak * lamda_expand)
        # output_region_out_gt = region_out_gt * (-c_leak * (input_expand - c_expand) - lamda_leak * lamda_expand)
        #
        # output_region_out_st_pos = output_region_out_st.clamp(min=zero_clip)
        # output_region_out_st_neg = output_region_out_st.clamp(max=-zero_clip)
        # output_region_out_st_all = output_region_out_st_pos + output_region_out_st_neg
        #
        # output_region_out_gt_pos = output_region_out_gt.clamp(min=zero_clip)
        # output_region_out_gt_neg = output_region_out_gt.clamp(max=-zero_clip)
        # output_region_out_gt_all = output_region_out_gt_pos + output_region_out_gt_neg


        output_region1 = lamda_valid_flag * region1_flag * 2 * ((1 - (torch.abs(input_expand - c_expand) / lamda_expand)) ** 2)
        output_region2 = lamda_valid_flag * region2_flag * (1 - 2 * ((torch.abs(input_expand - c_expand) / lamda_expand) ** 2))

        output_clip1_pos = output_region1.clamp(min=zero_clip)
        output_clip1_neg = output_region1.clamp(max=-zero_clip)
        output_clip1 = output_clip1_pos + output_clip1_neg

        output_clip2_pos = output_region2.clamp(min=zero_clip)
        output_clip2_neg = output_region2.clamp(max=-zero_clip)
        output_clip2 = output_clip2_pos + output_clip2_neg

        # output = (output_region1 + output_region2 +
        #           output_region_out_st + output_region_out_gt).prod(dim=1)
        # output = (output_region1 + output_region2).prod(dim=1)
        output = (output_clip1 + output_clip2).prod(dim=1)

        # norm_grad(output, 1)
        return output.clamp(min=zero_clip)

def norm_first_layer(*args): # input_expand, c_expand
    input_expand = args[0]
    c_expand = args[1]
    y = -((input_expand - c_expand) ** 2)
    return y

def norm_second_layer(*args): # lamda_expand
    lamda_expand = args[0]
    y = (2 * (lamda_expand ** 2))
    return y

def norm_third_layer(*args): # first, second
    first = args[0]
    second = args[1]
    y = (first/second).exp()
    return y

class Membership_norm(nn.Module):
    def __init__(self, feature, class_num, init_c=None, init_lamda=None, c_grad=True, lamda_grad=True):
        super(Membership_norm, self).__init__()

        if init_c is None:
            self.c = nn.Parameter(data=0.0*torch.ones((feature, class_num), dtype=torch.float),
                                  requires_grad=c_grad)
        else:
            assert len(init_c.shape) == 2 and \
                   init_c.shape[0] == feature and \
                   init_c.shape[1] == class_num
            self.c = nn.Parameter(data=init_c, requires_grad=c_grad)

        if init_lamda is None:
            self.lamda = nn.Parameter(data=1.0*torch.ones((feature, class_num), dtype=torch.float),
                                      requires_grad=lamda_grad)
        else:
            assert len(init_lamda.shape) == 2 and \
                   init_lamda.shape[0] == feature and \
                   init_lamda.shape[1] == class_num
            self.lamda = nn.Parameter(data=init_lamda, requires_grad=lamda_grad)

    def forward(self, input):
        # input shape: N*D*(w*h)
        # c shape: D*C
        # lamda shape : D*C
        # output: N*C*(w*h)

        # print(input.shape)
        # print(self.c.shape)
        # print(self.lamda.shape)

        N = input.shape[0]
        assert input.shape[1] == self.c.shape[0] and self.c.shape[0] == self.lamda.shape[0]
        D = input.shape[1]
        assert self.c.shape[1] == self.lamda.shape[1]
        C = self.c.shape[1]
        assert len(input.shape) == 3
        wh = input.shape[2]
        input_expand = input.unsqueeze(2).expand([N, D, C, wh])

        c_expand = self.c.unsqueeze(0).unsqueeze(3).expand_as(input_expand)
        lamda_expand = self.lamda.unsqueeze(0).unsqueeze(3).expand_as(input_expand)
        # print(input_expand.shape)
        out = (-((input_expand - c_expand)**2)/(2 * (lamda_expand**2))).exp()
        # first = checkpoint(norm_first_layer, input_expand, c_expand)
        # second = checkpoint(norm_second_layer, lamda_expand)
        # out = checkpoint(norm_third_layer, first, second)

        output = out.prod(dim=1)

        return output.clamp(min=zero_clip)


class Membership_freeze(nn.Module):
    def __init__(self):
        super(Membership_freeze, self).__init__()

    def forward(self, input, c, lamda):
        # input shape: N*D*(w*h)
        # c shape: D*C
        # lamda shape : D*C
        # output: N*C*(w*h)

        # print(input.shape)
        # print(self.c.shape)
        # print(self.lamda.shape)

        N = input.shape[0]
        assert input.shape[1] == c.shape[0] and c.shape[0] == lamda.shape[0]
        D = input.shape[1]
        assert c.shape[1] == lamda.shape[1]
        C = c.shape[1]
        assert len(input.shape) == 3
        wh = input.shape[2]
        input_expand = input.unsqueeze(2).expand([N, D, C, wh])

        c_expand = c.unsqueeze(0).unsqueeze(3).expand_as(input_expand)
        lamda_expand = lamda.unsqueeze(0).unsqueeze(3).expand_as(input_expand)

        out = (-((input_expand - c_expand)**2)/(2 * (lamda_expand**2))).exp()

        output = out.prod(dim=1)

        return output.clamp(min=zero_clip), out


import numpy as np

if __name__ == '__main__':
    from losses import FocalLoss, CenterLoss
    from networks.MemberShip_cuda import Membership_norm_cuda
    dim = 100

    x_in1 = np.linspace(-1, 1, num=dim)
    x_in2 = np.linspace(-1, 1, num=dim)
    x_in = 0.1*np.ones((dim*dim, 2), dtype=np.float)
    y = np.zeros((dim*dim, 4), dtype=np.float)

    for i, x1 in enumerate(x_in1):
        for j, x2 in enumerate(x_in2):
            x_in[i * dim + j][0] = x1
            x_in[i * dim + j][1] = x2
            if x1 > 0 and x2 > 0:
                y[i * dim + j][0] = np.sqrt((-4 * (x1 - 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 - 0.5) ** 2 + 1).clip(min=1e-5))
            elif x1 > 0 and x2 < 0:
                y[i * dim + j][1] = np.sqrt((-4 * (x1 - 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 + 0.5) ** 2 + 1).clip(min=1e-5))
            elif x1 < 0 and x2 < 0:
                y[i * dim + j][2] = np.sqrt((-4 * (x1 + 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 + 0.5) ** 2 + 1).clip(min=1e-5))
            elif x1 < 0 and x2 > 0:
                y[i * dim + j][3] = np.sqrt((-4 * (x1 + 0.5) ** 2 + 1).clip(min=1e-5) * (-4 * (x2 - 0.5) ** 2 + 1).clip(min=1e-5))

    # x_in = x_in1
    # y=(-4 * (x_in - 0.5) ** 2 + 1).clip(min=0.0)
    # print(x_in)
    # print(y)
    # exit()

    x_in_tensor = torch.tensor(x_in, requires_grad=False, dtype=torch.float).cuda()
    y_label_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float).unsqueeze(2).cuda()

    # layer = Membership_Activation(2, 4,
    #                               init_c=0.2*torch.ones((2, 4), dtype=torch.float),
    #                               init_lamda=1.5*torch.ones((2, 4), dtype=torch.float)).cuda()

    fc = torch.nn.Linear(2, 2).cuda()
    layer = Membership_norm(2, 4,
                            init_c=-5 * torch.ones((2, 4), dtype=torch.float),
                            init_lamda=4 * torch.ones((2, 4), dtype=torch.float)).cuda()

    # x = torch.tensor([[[0.9, 0.1], [0.9, 0.1]], [[-0.9, 0.1], [-0.1, -2.5]]], dtype=torch.float, requires_grad=True)
    # x2 = x ** 2
    # print(x2.requires_grad)
    # print(x.shape)
    # print(layer(x))
    # print(layer.c)
    # print(x.shape)
    # loss_focal = torch.nn.MSELoss()
    loss_focal = FocalLoss()
    loss_center = CenterLoss()
    para = [
        {"params": fc.parameters(), "lr": 1e-3},
        {"params": layer.c, "lr": 1e-3},
        {"params": layer.lamda, "lr": 1e-3},
    ]

    # optim = torch.optim.SGD(para)
    optim = torch.optim.Adam(para)
    # bestloss = 1e5
    # bestnetweightfc = []
    # bestnetweightlayer = []

    for i in range(0, 100000):
        h = fc(x_in_tensor).unsqueeze(2)
        y = layer(h)
        # print(y.squeeze(2))

        floss = loss_focal(y, y_label_tensor)
        closs = 0.5*loss_center(h, layer.c, y)
               # + 0.5*((0.0-layer.lamda).clamp(min=1e-5).sum()) \
               # - 0.5 * ((0.6 - layer.lamda).clamp(max=-1e-5).sum()) \
            # + 0.5*((layer.c-1).clamp(min=1e-5).sum())\
            # - 0.5*((1+layer.c).clamp(max=-1e-5).sum()) #\
        loss = floss + closs
        optim.zero_grad()
        loss.backward()
        # print(layer.c.grad)
        # print(layer.lamda.grad)
        optim.step()
        print(loss)
        # print(layer.lamda)
        # print(layer.c)

# import numpy as np
# if __name__ == '__main__':
#     layer = Membership_norm(2, 4,
#                             init_c=-5 * torch.ones((2, 4), dtype=torch.float),
#                             init_lamda=4 * torch.ones((2, 4), dtype=torch.float)).cuda()
#     layer.c = 0.5 * layer.c + 1


# if __name__ == '__main__':
#     import torch
#     class fc256_2_256(torch.nn.Module):
#         def __init__(self):
#             super(fc256_2_256, self).__init__()
#             self.layer1 = torch.nn.Linear(256, 2)
#             self.layer1_activation = torch.nn.ReLU()
#
#             self.layer2 = torch.nn.Linear(2, 256)
#             self.layer2_activation = torch.nn.ReLU()
#
#         def forward(self, input_tensor):
#             h = self.layer1(input_tensor)
#             h = self.layer1_activation(h)
#             h = self.layer2(h)
#             out = self.layer2_activation(h)
#             return out
#     model = fc256_2_256()
#     data = torch.rand(2000, 256)
#     label = torch.rand(2000, 256)
#     Loss = torch.nn.MSELoss()
#     opt = torch.optim.SGD(model.parameters(), lr=1e-2)
#
#     for i in range(0, 100):
#         y = model(data)
#         loss = Loss(y, label)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         print(loss)


# import numpy as np
# if __name__ == '__main__':
#     from losses import FocalLoss
#     x_in1 = np.linspace(-1, 1, num=7)
#     y=(-4 * (x_in1 - 0.5) ** 2 + 1).clip(min=0.0)
#     print(x_in1)
#     print(y)
#     x_in_tensor = torch.tensor(x_in1, requires_grad=False, dtype=torch.float).unsqueeze(1).unsqueeze(2)
#     y_label_tensor = torch.tensor(y, requires_grad=False, dtype=torch.float).unsqueeze(1).unsqueeze(2)
#
#     layer = Membership_norm(1, 1,
#                                 init_c=0.8*torch.ones((1, 1), dtype=torch.float),
#                                 init_lamda=0.8*torch.ones((1, 1), dtype=torch.float))
#
#     loss_fun = torch.nn.MSELoss()
#     # loss_fun = FocalLoss()
#     para = [
#         {"params": layer.c, "lr": 1e-3},
#         {"params": layer.lamda, "lr": 1e-3},
#     ]
#
#     optim = torch.optim.SGD(para)
#     for i in range(0, 10000):
#         y = layer(x_in_tensor)
#         # print(y.squeeze(2))
#         # print(y)
#         # ((y_label_tensor[0]-y[0])**2).backward()
#         # loss_fun(y_label_tensor[4], y[4]).backward()
#         # print(layer.lamda.grad)
#         # exit()
#
#         loss = loss_fun(y, y_label_tensor)
#         optim.zero_grad()
#         loss.backward()
#         # print(layer.lamda.grad)
#         optim.step()
#         print(layer.lamda)
#         print(layer.c)
