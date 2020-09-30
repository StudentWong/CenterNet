from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

try:
    from .DCNv2.dcn_v2 import DCN
except:
    from src.lib.models.networks.DCNv2.dcn_v2 import DCN
    pass
try:
    from ..membership import Membership_Activation, Membership_norm
except:
    from src.lib.models.membership import Membership_Activation, Membership_norm
    pass


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        #if name.endswith('.pth'):
        #    model_weights = torch.load(data + name)
        #else:
        #    model_url = get_model_url(data, name, hash)
        #    model_weights = model_zoo.load_url(model_url)
        model_weights = torch.load('../weights/dla34-ba72cf86.pth')

        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model

class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])  
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
     
            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, 
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)
                 
        
    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])



class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) -i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg_no_bias(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):

        super(DLASeg_no_bias, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        # print(pretrained)
        self.base = globals()[base_name](pretrained=pretrained)
        #base = dla34(pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:

              if 'hm' in head:
                fc = nn.Sequential(
                      nn.Conv2d(channels[self.first_level], head_conv,
                                kernel_size=3, padding=1, bias=True))
                # fc = nn.Sequential(
                #     nn.Conv2d(channels[self.first_level], head_conv,
                #               kernel_size=3, padding=1, bias=True),
                #     nn.ReLU(inplace=True),
                #     nn.Conv2d(head_conv, head_conv,
                #               kernel_size=final_kernel, stride=1,
                #               padding=final_kernel // 2, bias=True))
                # fc = nn.Sequential(
                #           nn.Conv2d(channels[self.first_level], head_conv,
                #                     kernel_size=3, padding=1, bias=True),
                #           nn.ReLU(inplace=True),
                #           nn.Conv2d(head_conv, classes,
                #                     kernel_size=final_kernel, stride=1,
                #                     padding=final_kernel // 2, bias=True))
                fc[-1].bias.data.fill_(0.0)
              else:
                fc = nn.Sequential(
                      nn.Conv2d(channels[self.first_level], head_conv,
                                kernel_size=3, padding=1, bias=True),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(head_conv, classes,
                                kernel_size=final_kernel, stride=1,
                                padding=final_kernel // 2, bias=True))
                fill_fc_weights(fc)
            else:
              # fc = nn.Conv2d(channels[self.first_level], classes,
              #     kernel_size=final_kernel, stride=1,
              #     padding=final_kernel // 2, bias=True)
              fc = nn.Conv2d(channels[self.first_level], head_conv,
                  kernel_size=final_kernel, stride=1,
                  padding=final_kernel // 2, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(0.0)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)


        # c1 = [2.0, -5.0, -3.5, -6.0, -3.0]
        # lamda1 = [2.0, 3.0, 2.5, 6.0, 4.0]
        # c2 = [-5.0, 1.5, -2.5, -6.0, -5.0]
        # lamda2 = [3.0, 1.5, 3.5, 3.0, 2.0]
        # c3 = [-4.0, -3.5, 0.0, -6.0, -5.0]
        # lamda3 = [4.0, 3.5, 2.0, 4.0, 3.0]
        # c4 = [-3.5, -4.5, -5.0, 1.0, -3.5]
        # lamda4 = [3.5, 3.5, 2.0, 3.0, 3.5]
        # c5 = [-4.5, -6.5, -4.5, -5.0, 1.0]
        # lamda5 = [3.5, 3.5, 3.5, 5.0, 3.0]


        # c = [c1, c2, c3, c4, c5]
        # lamda = [lamda1, lamda2, lamda3, lamda4, lamda5]


        # init_c = torch.tensor(np.array(c), dtype=torch.float)
        # init_lamda = torch.tensor(np.array(lamda), dtype=torch.float)

        # self.menber_activation = Membership_Activation(5, 5,
        #                                                init_c=init_c,
        #                                                init_lamda=init_lamda)

        # self.menber_activation = Membership_norm(5, 5,
        #                                          init_c=init_c/1,
        #                                          init_lamda=init_lamda/1)
        self.menber_activation = Membership_norm(head_conv, heads['hm'])
        # self.menber_activation = Membership_norm(5, 5)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:

            if head == 'hm':
                # print(z[head])
                z['ft'] = self.__getattr__(head)(y[-1])
                origin_shape = z['ft'].shape
                z[head] = self.menber_activation(
                    z['ft'].view(origin_shape[0], origin_shape[1], origin_shape[2]*origin_shape[3])
                ).view(origin_shape[0], 5, origin_shape[2], origin_shape[3])
                # print(z[head])
                z['center'] = self.menber_activation.c
            else:
                z[head] = self.__getattr__(head)(y[-1])
        return [z]
    

def get_pose_net_gt_centerloss(num_layers, heads, head_conv=256, down_ratio=4):
  model = DLASeg_no_bias('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model

# import cv2
# if __name__ == '__main__':
#     from src.lib.models.networks.DCNv2.dcn_v2 import DCN
#     #num_layers: 34
#     #heads: {'hm': 5, 'wh': 2, 'reg': 2}
#     #head_conv: 256
#     model = get_pose_net_no_bias(num_layers=34, heads={'hm': 5, 'wh': 2, 'reg': 2},
#                                  head_conv=256)
#     # print(model)
#     ckpt = torch.load('/home/studentw/disk3/tracker/CenterNet/exp/ctdet/default/model_best_shangqi.pth')
#     # print(ckpt['state_dict'].keys())
#     model.load_state_dict(ckpt['state_dict'])
#     model = model.cuda()
#
#     mean = np.array([0.40789654, 0.44719302, 0.47026115],
#                     dtype=np.float32).reshape(1, 1, 3)
#     std = np.array([0.28863828, 0.27408164, 0.27809835],
#                    dtype=np.float32).reshape(1, 1, 3)
#
#     img = cv2.imread('/home/studentw/disk3/shangqi/train/200420000000.png')/255.0
#
#
#     inp = (img - mean) / std
#     inp = inp.transpose(2, 0, 1)
#
#     # print(img-mean)
#     input = torch.tensor(inp, dtype=torch.float).unsqueeze(0).cuda()
#     y = model(input)
#     # print(y[0].keys())
#     print(np.max(y[0]['hm'][0][0:1].sigmoid().permute(1, 2, 0).detach().cpu().numpy()))
#     print(np.min(y[0]['hm'][0][0:1].sigmoid().permute(1, 2, 0).detach().cpu().numpy()))
#
#     print(np.max(y[0]['hm'][0][0:1].permute(1, 2, 0).detach().cpu().numpy()))
#     print(np.min(y[0]['hm'][0][0:1].permute(1, 2, 0).detach().cpu().numpy()))
#     cv2.imshow('1', y[0]['hm'][0][0:1].permute(1, 2, 0).detach().cpu().numpy())
#     cv2.waitKey(0)


def save_features_output(ckpt_name, involve_train=False):
    # from src.lib.models.networks.DCNv2.dcn_v2 import DCN
    import cv2
    from pycocotools import coco
    from src.lib.models.decode import ctdet_decode_ret_peak
    import seaborn as sns
    import pandas as pd

    # num_layers: 34
    # heads: {'hm': 5, 'wh': 2, 'reg': 2}
    # head_conv: 256
    model = get_pose_net_no_bias(num_layers=34, heads={'hm': 5, 'wh': 2, 'reg': 2},
                                 head_conv=256)
    # print(model)
    column = list(range(0, 256)) + ['class'] + ['score']
    df = pd.DataFrame(columns=column)
    df_line = 0
    # print(df)
    # exit()
    ckpt = torch.load('/home/studentw/disk3/tracker/CenterNet/exp/ctdetnfs/default/' + ckpt_name)
    # print(ckpt['state_dict'].keys())
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()

    # mean = np.array([0.40789654, 0.44719302, 0.47026115],
    #                 dtype=np.float32).reshape(1, 1, 3)
    # std = np.array([0.28863828, 0.27408164, 0.27809835],
    #                dtype=np.float32).reshape(1, 1, 3)
    mean = np.array([0.317200417, 0.317200417, 0.317200417],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.22074733, 0.22074733, 0.22074733],
                   dtype=np.float32).reshape(1, 1, 3)

    all_img = []

    if involve_train:
        train_data_dir = '/home/studentw/disk3/tracker/CenterNet/data/shangqi/train'
        train_list = sorted(os.listdir(train_data_dir))
        train_num = len(train_list)
        for img_name in train_list:
            all_img = all_img + [os.path.join(train_data_dir, img_name)]
        label_train = coco.COCO('/home/studentw/disk3/tracker/CenterNet/data/shangqi/annotations/train.json')

    val_data_dir = '/home/studentw/disk3/tracker/CenterNet/data/shangqi/val'
    val_list = sorted(os.listdir(val_data_dir))
    val_num = len(val_list)
    for img_name in val_list:
        all_img = all_img + [os.path.join(val_data_dir, img_name)]

    label_val = coco.COCO('/home/studentw/disk3/tracker/CenterNet/data/shangqi/annotations/val.json')

    mapping_ids = [1, 2, 3, 7, 8]
    invert_mapping = {1: 0, 2: 1, 3: 2, 7: 3, 8: 4}
    output_file = []
    feature_res = []
    # for clsnum in range(0, 5):
    #     file_name = '/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out{:d}.npy'.format(clsnum)
    #     if not os.path.exists(file_name):
    #         os.mknod(file_name)
    #     output_file = output_file + [file_name]
    #     feature_res = feature_res + [[]]
    # print(feature_res)

    all_num = len(all_img)

    for i, img_path in enumerate(all_img):
        imgnum_in_json = int(img_path.split('/')[-1].replace('.png', ''))
        # print(imgnum_in_json)
        # exit()
        if involve_train and i <= train_num-1:
            label_id = label_train.getAnnIds(imgIds=[imgnum_in_json])
            # print(label_id)
            labels = [label_train.anns[label_id_single] for label_id_single in label_id]
            # print(labels)
        else:
            label_id = label_val.getAnnIds(imgIds=[imgnum_in_json])
            labels = [label_val.anns[label_id_single] for label_id_single in label_id]

        img = cv2.imread(img_path) / 255.0

        inp = (img - mean) / std
        inp = inp.transpose(2, 0, 1)

        # print(img-mean)
        input = torch.tensor(inp, dtype=torch.float).unsqueeze(0).cuda()
        with torch.no_grad():
            y = model(input)[-1]

        # print(np.max(y['reg'][0][:].permute(1, 2, 0).detach().cpu().numpy()))
        # exit()
        dets, xs, ys = ctdet_decode_ret_peak(y['hm'], y['wh'], reg=y['reg'], cat_spec_wh=False, K=20)
        det_box_tlxywh = dets[0, :, 0:4].detach().cpu().numpy() * 4

        det_box_cwh = det_box_tlxywh.copy()
        det_box_cwh[:, 2:4] = det_box_tlxywh[:, 2:4] - det_box_tlxywh[:, 0:2]
        det_box_cwh[:, 0:2] = det_box_tlxywh[:, 0:2] + 0.5*det_box_cwh[:, 2:4]

        for label in labels:
            matched = False
            lab_cwh = np.array(label['bbox'], dtype=np.float)
            lab_cwh[2:4] = np.array(label['bbox'][2:4])
            lab_cwh[0:2] = np.array(label['bbox'][0:2]) + 0.5 * lab_cwh[2:4]

            for detnum, det in enumerate(det_box_cwh):
                # print(lab_cwh)
                distance_xy = (lab_cwh[0:2]-det[0:2]) ** 2
                if np.sqrt(distance_xy[0]+distance_xy[1])<2 \
                        and mapping_ids[int(dets[0][detnum][5])] == label['category_id']:
                    # print(int(lab_cwh[0]/4))
                    # print(xs[0][detnum])
                    # print(y['reg'][0, 1, int(ys[0][detnum]), int(xs[0][detnum])])
                    # print(int(lab_cwh[1] / 4))
                    # print(ys[0][detnum])
                    # print(y['reg'][0, 0, int(ys[0][detnum]), int(xs[0][detnum])])
                    # print(y['hm'][0, int(dets[0][detnum][5]), int(ys[0][detnum]), int(xs[0][detnum])])
                    # print('end')

                    feature_np = y['ft'][0, :, int(ys[0][detnum]),int(xs[0][detnum])].detach().cpu().numpy()
                    matched = True
                    feature_dict = dict()
                    # print(feature_np)
                    for fnum, f in enumerate(feature_np):
                        feature_dict[fnum] = f
                    feature_dict['class'] = label['category_id']
                    feature_dict['score'] = float(y['hm'][
                                                    0, invert_mapping[label['category_id']], int(ys[0][detnum]), int(
                                                    xs[0][detnum])].detach().cpu().numpy())
                    df.loc[df_line] = feature_dict
                    df_line = df_line + 1


            if not matched and (label['category_id'] in mapping_ids):
                feature_np = y['ft'][0, :, int(lab_cwh[1] / 4), int(lab_cwh[0] / 4)].detach().cpu().numpy()
                matched = True
                feature_dict = dict()
                # print(feature_np)
                for fnum, f in enumerate(feature_np):
                    feature_dict[fnum] = f
                feature_dict['class'] = label['category_id']
                feature_dict['score'] = float(y['hm'][
                                                  0, invert_mapping[label['category_id']], int(lab_cwh[1] / 4), int(lab_cwh[0] / 4)].detach().cpu().numpy())
                df.loc[df_line] = feature_dict
                df_line = df_line + 1
        print('{:d}/{:d}'.format(i, all_num))
        # print(df)
        # exit()
    df.to_csv("./" + ckpt_name.replace('.pth', '') + '.csv')

def draw_box(path, feature=256):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    heads = list(range(0, feature))
    df = pd.read_csv(path, header=0,index_col=0)
    # print(df)
    for head in heads:
        sns.boxplot(x="class", y=str(head), data=df)
        # sns.swarmplot(x="class", y=str(head), data=df, color=".25")
        plt.show()

def draw_reduce_dim_feature(path, feature=256, transformer=None, alpha=False):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    cat_id = [1, 2, 3, 7, 8]
    heads = list(range(0, feature))
    color_mapping = {1: 'red', 2: 'blue', 3: 'green', 7: 'orange', 8: 'black'}
    df = pd.read_csv(path, header=0, index_col=0)
    heads_feature = list(range(0, 256))
    for i, f_int in enumerate(heads_feature):
        heads_feature[i] = str(f_int)

    feat_np = np.array(df[heads_feature])
    cat_np = np.array(df['class'])
    score = np.array(df['score'])

    color = []
    for cat_i in cat_np:
        color = color + [color_mapping[cat_i]]
    # print(cat_np)
    if transformer is None:
        transformer = PCA(2)
        transformer.fit(feat_np)
    reduce_dim = transformer.transform(feat_np)
    # plt.scatter(reduce_dim[:, 0], reduce_dim[:, 1], c=color, s=3, alpha=score)
    all_len = reduce_dim.shape[0]
    if alpha:
        for i in range(0, all_len):
            plt.scatter(reduce_dim[i:i+1, 0], reduce_dim[i:i+1, 1], c=color[i], s=3, alpha=score[i])
            print('{:d}/{:d}'.format(i, all_len))
    else:
        plt.scatter(reduce_dim[:, 0], reduce_dim[:, 1], c=color, s=3)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
    return transformer
    # print(reduce_dim.shape)
    # for cat in cat_id:



# from src.lib.models.networks.DCNv2.dcn_v2 import DCN
# if __name__ == '__main__':
#     class_feature0 = np.load('/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out0.npy')
#     class_feature1 = np.load('/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out1.npy')
#     class_feature2 = np.load('/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out2.npy')
#     class_feature3 = np.load('/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out3.npy')
#     class_feature4 = np.load('/home/studentw/disk3/tracker/CenterNet/data/shangqi/feature_out4.npy')
#     import matplotlib.pyplot as plt
#     class_feature = [class_feature0, class_feature1, class_feature2, class_feature3, class_feature4]
#     color = ['red', 'blue', 'green', 'yellow', 'hotpink']
#
#     for i in range(0, 5):
#         figure = plt.figure(i)
#         for j in range(0, 5):
#             x = np.sqrt(class_feature[j][:, 5])
#             y = class_feature[j][:, i]
#             plt.scatter(x, y, c=color[j], s=8, label=j, alpha=0.6, edgecolors='gray', linewidths=0.5)
#         plt.show()


# if __name__ == '__main__':
#     from src.lib.models.membership import Membership_Activation
#     from src.lib.models.networks.DCNv2.dcn_v2 import DCN
#     model = get_pose_net_no_bias(num_layers=34, heads={'hm': 5, 'wh': 2, 'reg': 2},
#                                  head_conv=256)



if __name__ == '__main__':
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt


    # testdata_cat1 = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    # testdata_cat2 = [[1, 1], [1.5, 1], [2, 1], [2.5, 1], [3, 1]]
    # df = np.array(testdata_cat1 + testdata_cat2)
    # df = pd.DataFrame(columns=['data', 'class'])
    # print(df)
    # df.loc[0]={'data':0, 'class':1}
    # df.loc[1] = {'data':0.5, 'class':2}
    # print(df)
    # exit()
    # sns.boxplot(x="class", y="data", data=df)
    # plt.show()
    # save_features_output('2080_lamda0.05_batch8_lr1.25e-4_ap82.pth')

    # draw_box('/home/studentw/disk3/tracker/CenterNet/src/2080_lamda0.01_batch8_lr1.25e-4_ap83_best.csv')
    draw_reduce_dim_feature('/home/studentw/disk3/tracker/CenterNet/src/2080_lamda0.01_batch8_lr1.25e-4_ap83_best.csv')
    draw_reduce_dim_feature('/home/studentw/disk3/tracker/CenterNet/src/2080_lamda0.05_batch8_lr1.25e-4_ap82.csv')



