# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN
from .dynamic_conv import Dynamic_conv2d
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=1, bias=False, K=8)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def get_att(xR, xT):
    assert xR.shape == xT.shape
    N = xR.shape[0]
    C = xR.shape[1]
    W = xR.shape[2]
    H = xR.shape[3]
    xR_O_N_WH_C = xR.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
    xR_O_N_C_WH = xR_O_N_WH_C.permute([0, 2, 1]).contiguous()
    xR_O_self_NCC = torch.bmm(xR_O_N_C_WH, xR_O_N_WH_C)

    xT_O_N_WH_C = xT.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
    xT_O_N_C_WH = xT_O_N_WH_C.permute([0, 2, 1]).contiguous()
    xT_O_self_NCC = torch.bmm(xT_O_N_C_WH, xT_O_N_WH_C)

    xR_O_self_NCC = (xR_O_self_NCC-xR_O_self_NCC.mean())/xR_O_self_NCC.std()
    xT_O_self_NCC = (xT_O_self_NCC - xT_O_self_NCC.mean()) / xT_O_self_NCC.std()

    return xR_O_self_NCC, xT_O_self_NCC


class Att_fusion_final(nn.Module):
    def __init__(self, inplanes):
        super(Att_fusion_final, self).__init__()
        self.adjust_convR = nn.Conv2d(inplanes, inplanes,
                               kernel_size=1, stride=1)
        self.adjust_convT = nn.Conv2d(inplanes, inplanes,
                               kernel_size=1, stride=1)

    def forward(self, xR, xT):
        assert xR.shape == xT.shape
        N = xR.shape[0]
        C = xR.shape[1]
        W = xR.shape[2]
        H = xR.shape[3]

        xR_O_N_WH_C = xR.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        # xR_O_N_C_WH = xR_O_N_WH_C.permute([0, 2, 1]).contiguous()
        # xR_O_self_NCC = torch.bmm(xR_O_N_C_WH, xR_O_N_WH_C).unsqueeze(3)

        xT_O_N_WH_C = xT.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        # xT_O_N_C_WH = xT_O_N_WH_C.permute([0, 2, 1]).contiguous()
        # xT_O_self_NCC = torch.bmm(xT_O_N_C_WH, xT_O_N_WH_C).unsqueeze(3)

        xR_a = self.adjust_convR(xR)
        xT_a = self.adjust_convT(xT)
        assert xR.shape == xT.shape == (N, C, W, H)
        # print(N)
        # print(C)
        # print(W)
        # print(H)

        xR_N_WH_C = xR_a.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        xR_N_C_WH = xR_N_WH_C.permute([0, 2, 1]).contiguous()
        xR_self_NCC = torch.bmm(xR_N_C_WH, xR_N_WH_C).unsqueeze(3)

        xT_N_WH_C = xT_a.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        xT_N_C_WH = xT_N_WH_C.permute([0, 2, 1]).contiguous()
        xT_self_NCC = torch.bmm(xT_N_C_WH, xT_N_WH_C).unsqueeze(3)

        # att = torch.cat([xR_self_NCC, xT_self_NCC], dim=3)
        # att = torch.softmax(att, dim=3)
        #
        # xR_self_NCC, xT_self_NCC = torch.split(att, [1, 1], dim=3)
        # xR_self_NCC = xR_self_NCC.squeeze(3)
        # xT_self_NCC = xT_self_NCC.squeeze(3)
        #
        # # xR_self_NCC = xR_self_NCC.squeeze(3).sigmoid()
        # # xT_self_NCC = xT_self_NCC.squeeze(3).sigmoid()
        #
        # xR_new = torch.bmm(xR_O_N_WH_C, xR_self_NCC)
        # xT_new = torch.bmm(xT_O_N_WH_C, xT_self_NCC)
        #
        # x = xR_new + xT_new

        xR_self_NCC = xR_self_NCC.squeeze(3)
        xT_self_NCC = xT_self_NCC.squeeze(3)
        x_self_N2CC = torch.cat([xR_self_NCC, xT_self_NCC], dim=1).softmax(dim=1)
        x_O_N_WH_2C = torch.cat([xR_O_N_WH_C, xT_O_N_WH_C], dim=2)
        # print(x_self_N2CC)
        # exit()
        x = torch.bmm(x_O_N_WH_2C, x_self_N2CC)
        return x.view(N, W, H, C).permute([0, 3, 1, 2]).contiguous()

class fusion_softmax(nn.Module):
    def __init__(self, inplanes):
        super(fusion_softmax, self).__init__()
        self.adjust_convR = nn.Conv2d(inplanes, inplanes,
                               kernel_size=1, stride=1)
        self.adjust_convT = nn.Conv2d(inplanes, inplanes,
                               kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(inplanes*2, momentum=BN_MOMENTUM, affine=True)

    def forward(self, xR, xT):
        assert xR.shape == xT.shape
        N = xR.shape[0]
        C = xR.shape[1]
        W = xR.shape[2]
        H = xR.shape[3]

        xR_a = self.adjust_convR(xR)
        xT_a = self.adjust_convT(xT)
        cat_a = self.bn(torch.cat([xR_a, xT_a], dim=1))
        xR_a, xT_a = torch.split(cat_a, [C, C], dim=1)

        assert xR.shape == xT.shape == (N, C, W, H)

        xR_a = xR_a.unsqueeze(4)
        xT_a = xT_a.unsqueeze(4)

        att = torch.cat([xR_a, xT_a], dim=4)
        att = torch.softmax(att, dim=4)

        xR_a, xT_a = torch.split(att, [1, 1], dim=4)
        xR_a = xR_a.squeeze(4)
        xT_a = xT_a.squeeze(4)

        xR_new = xR * xR_a
        xT_new = xT * xT_a
        # xR_new = xR * 0.5
        # xT_new = xT * 0.5
        x = xR_new + xT_new

        # print(xR_new.view(N, W, H, C).permute([0, 3, 1, 2]))
        # print(xR[0, 0, 0, :])
        # print(xR_O_N_WH_C.view(N, W, H, C).permute([0, 3, 1, 2])[0, 0, 0, :])
        return x

# class fusion_pooling(nn.Module):
#     def __init__(self, inplanes):
#         super(Att_fusion_softmax, self).__init__()
#         self.adjust_convR = nn.Conv2d(inplanes, inplanes,
#                                kernel_size=1, stride=1)
#         self.adjust_convT = nn.Conv2d(inplanes, inplanes,
#                                kernel_size=1, stride=1)
#
#     def forward(self, xR, xT):
#         assert xR.shape == xT.shape
#         N = xR.shape[0]
#         C = xR.shape[1]
#         W = xR.shape[2]
#         H = xR.shape[3]
#
#         xR_O_N_WH_C = xR.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
#         # xR_O_N_C_WH = xR_O_N_WH_C.permute([0, 2, 1]).contiguous()
#         # xR_O_self_NCC = torch.bmm(xR_O_N_C_WH, xR_O_N_WH_C).unsqueeze(3)
#
#         xT_O_N_WH_C = xT.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
#         # xT_O_N_C_WH = xT_O_N_WH_C.permute([0, 2, 1]).contiguous()
#         # xT_O_self_NCC = torch.bmm(xT_O_N_C_WH, xT_O_N_WH_C).unsqueeze(3)
#
#         xR_a = self.adjust_convR(xR)
#         xT_a = self.adjust_convT(xT)
#         assert xR.shape == xT.shape == (N, C, W, H)
#
#         xR_N_WH_C = xR_a.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
#         xR_N_C_WH = xR_N_WH_C.permute([0, 2, 1]).contiguous()
#         xR_self_NCC = torch.bmm(xR_N_C_WH, xR_N_WH_C).unsqueeze(3)
#
#         xT_N_WH_C = xT_a.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
#         xT_N_C_WH = xT_N_WH_C.permute([0, 2, 1]).contiguous()
#         xT_self_NCC = torch.bmm(xT_N_C_WH, xT_N_WH_C).unsqueeze(3)
#
#         att = torch.cat([xR_self_NCC, xT_self_NCC], dim=3)
#         att = torch.softmax(att, dim=3)
#
#         xR_self_NCC, xT_self_NCC = torch.split(att, [1, 1], dim=3)
#         xR_self_NCC = xR_self_NCC.squeeze(3)
#         xT_self_NCC = xT_self_NCC.squeeze(3)
#         #
#         # # xR_self_NCC = xR_self_NCC.squeeze(3).sigmoid()
#         # # xT_self_NCC = xT_self_NCC.squeeze(3).sigmoid()
#         #
#         xR_new = torch.bmm(xR_O_N_WH_C, xR_self_NCC)
#         xT_new = torch.bmm(xT_O_N_WH_C, xT_self_NCC)
#
#         x = xR_new + xT_new
#         return x.view(N, W, H, C).permute([0, 3, 1, 2]).contiguous()

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1_R = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_T = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)

        self.bn1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.att_fusion = fusion_softmax(512)
        # self.att_fusion = Att_fusion_final(512)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def update_temp_layer(self, layers):
        for layer in layers:
            # print(isinstance(layer, BasicBlock))
            if isinstance(layer, BasicBlock):
                layer.conv1.update_temperature()
                layer.conv2.update_temperature()
                if layer.downsample is not None:
                    layer.downsample[0].update_temperature()
            else:
                print(layer)
                exit()

    def set_temp_layer(self, layers, temp):
        for layer in layers:
            # print(isinstance(layer, BasicBlock))
            if isinstance(layer, BasicBlock):
                layer.conv1.set_temperature(1)
                layer.conv2.set_temperature(1)
                if layer.downsample is not None:
                    layer.downsample[0].set_temperature(1)
            else:
                print(layer)
                exit()
        # exit()
    def update_temp(self):
        self.update_temp_layer(self.layer1)
        self.update_temp_layer(self.layer2)
        self.update_temp_layer(self.layer3)
        self.update_temp_layer(self.layer4)

    def set_temp(self, temp):
        self.set_temp_layer(self.layer1, temp)
        self.set_temp_layer(self.layer2, temp)
        self.set_temp_layer(self.layer3, temp)
        self.set_temp_layer(self.layer4, temp)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     nn.Conv2d(self.inplanes, planes * block.expansion,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            # )
            downsample = nn.Sequential(
                    Dynamic_conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False, K=8),
                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        xR = x[:, 0:3, :, :]
        xT = x[:, 3:4, :, :].expand_as(xR)

        xR = self.conv1_R(xR)
        xT = self.conv1_T(xT)
        x = torch.cat([xR, xT], dim=1).contiguous()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        # exit()
        sum_dim = x.shape[1]
        xR, xT = torch.split(x, [int(sum_dim/2), int(sum_dim/2)], dim=1)
        xR = xR.contiguous()
        xT = xT.contiguous()
        # print(xR.shape)
        # print(xT.shape)
        # exit()
        ret = {}
        ret['xR_att'] = []
        ret['xT_att'] = []
        xR_att, xT_att = get_att(xR, xT)
        ret['xR_att'] = ret['xR_att'] + [xR_att]
        ret['xT_att'] = ret['xT_att'] + [xT_att]

        xR = self.layer1(xR)
        xT = self.layer1(xT)
        xR_att, xT_att = get_att(xR, xT)
        ret['xR_att'] = ret['xR_att'] + [xR_att]
        ret['xT_att'] = ret['xT_att'] + [xT_att]

        xR = self.layer2(xR)
        xT = self.layer2(xT)
        xR_att, xT_att = get_att(xR, xT)
        ret['xR_att'] = ret['xR_att'] + [xR_att]
        ret['xT_att'] = ret['xT_att'] + [xT_att]

        xR = self.layer3(xR)
        xT = self.layer3(xT)
        xR_att, xT_att = get_att(xR, xT)
        ret['xR_att'] = ret['xR_att'] + [xR_att]
        ret['xT_att'] = ret['xT_att'] + [xT_att]

        xR = self.layer4(xR)
        xT = self.layer4(xT)
        xR_att, xT_att = get_att(xR, xT)
        ret['xR_att'] = ret['xR_att'] + [xR_att]
        ret['xT_att'] = ret['xT_att'] + [xT_att]

        x = self.att_fusion(xR, xT)
        # x = xR + xT
        x = self.deconv_layers(x)

        for head in self.heads:
            if head == 'hm':
                xx = self.__getattr__(head)[0](x)
                ret['ft'] = self.__getattr__(head)[1](xx)
                # print(self.__getattr__(head).__len__())
                # exit()
                ret[head] = self.__getattr__(head)[2](ret['ft'])
            else:
                ret[head] = self.__getattr__(head)(x)
        return [ret]

    def load_state_dict_fusion(self, statdict):
        ckpt_keys = statdict.keys()

        for key in ckpt_keys:
            key_split = key.split(".")
            if len(key_split) == 4:
                if key_split[0].startswith("layer") \
                        and key_split[2].startswith("conv") \
                        and key_split[3] == "weight":
                    new_ckpt = statdict[key].unsqueeze(0).expand(self.state_dict()[key].shape)
                    statdict[key] = new_ckpt

            elif len(key_split) == 5:
                if key_split[0].startswith("layer") \
                        and key_split[2] == "downsample" \
                        and key_split[3] == "0" \
                        and key_split[4] == "weight":
                    # print(key)
                    new_ckpt = statdict[key].unsqueeze(0).expand(self.state_dict()[key].shape)
                    statdict[key] = new_ckpt
            # layer2
            # .0.downsample
            # .0.weight
        # print(self.state_dict().keys())
        # print(ckpt_keys)
        # print(self.state_dict()['layer1.0.conv1.weight'].shape)
        # print(statdict['layer1.0.conv1.weight'].shape)
        # exit()
        self.load_state_dict(statdict, strict=False)

    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            # print(self.state_dict().keys())
            # exit()
            pretrained_state_dict = torch.load('../weights/resnet18fusion.pth')
            # print('=> loading pretrained model {}'.format(url))
            self.load_state_dict_fusion(pretrained_state_dict)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model
