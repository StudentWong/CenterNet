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
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
attention = False
affine = True

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_share(in_planes, out_planes, stride=1, ratio=0.5):
    """3x3 convolution with padding"""
    planes_share = int(out_planes * ratio)
    planes_spec = out_planes - planes_share

    R = nn.Conv2d(in_planes, planes_spec, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    T = nn.Conv2d(in_planes, planes_spec, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    if planes_share > 0:
        S = nn.Conv2d(in_planes, planes_share, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        return R, T, S
    else:
        return R, T, None
    # ret = dict()
    # ret["R"] = R
    # ret["T"] = T
    # ret["S"] = S


def bn_share(planes, affine, ratio=0.5):
    planes_share = int(planes * ratio)
    planes_spec = planes - planes_share
    R = nn.BatchNorm2d(planes_spec, momentum=BN_MOMENTUM, affine=affine)
    T = nn.BatchNorm2d(planes_spec, momentum=BN_MOMENTUM, affine=affine)
    if planes_share > 0:
        S = nn.BatchNorm2d(planes_share, momentum=BN_MOMENTUM, affine=affine)
        return R, T, S
    else:
        return R, T, None

class downsample_share(nn.Module):
    def __init__(self, inplanes, planes, block_expansion, stride, ratio=0.5):
        super(downsample_share, self).__init__()
        self.ratio = ratio
        # print(planes)
        # print(self.ratio)
        planes_share = int(planes * ratio)
        planes_spec = planes - planes_share
        self.planes_share = planes_share
        # print(planes_spec)
        self.convR = nn.Conv2d(inplanes, planes_spec * block_expansion,
                          kernel_size=1, stride=stride, bias=False)
        self.convT = nn.Conv2d(inplanes, planes_spec * block_expansion,
                               kernel_size=1, stride=stride, bias=False)
        if planes_share>0:
            self.convS = nn.Conv2d(inplanes, planes_share * block_expansion,
                               kernel_size=1, stride=stride, bias=False)
        self.bnR = nn.BatchNorm2d(planes_spec * block_expansion, momentum=BN_MOMENTUM, affine=affine)
        self.bnT = nn.BatchNorm2d(planes_spec * block_expansion, momentum=BN_MOMENTUM, affine=affine)
        if planes_share > 0:
            self.bnS = nn.BatchNorm2d(planes_share * block_expansion, momentum=BN_MOMENTUM, affine=affine)

    def forward(self, x):
        xR = x[0]
        xT = x[1]
        outR = self.convR(xR)
        outR = self.bnR(outR)
        if self.planes_share > 0:
            outS_R = self.convS(xR)
            outS_R = self.bnS(outS_R)

        outT = self.convT(xT)
        outT = self.bnT(outT)
        if self.planes_share > 0:
            outS_T = self.convS(xT)
            outS_T = self.bnS(outS_T)

        if self.planes_share > 0:
            ret_R = torch.cat([outR, outS_R], dim=1)
            ret_T = torch.cat([outT, outS_T], dim=1)
        else:
            ret_R = outR
            ret_T = outT
        return [ret_R, ret_T]



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine)
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


class BasicBlock_share(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, ratio=0.5):
        super(BasicBlock_share, self).__init__()
        self.share_ratio = ratio
        self.conv1_R, self.conv1_T, self.conv1_S = conv3x3_share(inplanes, planes, stride, ratio=ratio)
        self.bn1_R, self.bn1_T, self.bn1_S = bn_share(planes, affine=affine, ratio=ratio)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_R, self.conv2_T, self.conv2_S = conv3x3_share(planes, planes, ratio=ratio)
        self.bn2_R, self.bn2_T, self.bn2_S = bn_share(planes, affine=affine, ratio=ratio)
        self.downsample = downsample
        self.stride = stride
        # print(self.downsample)
        # print(self.conv2_R)
        # print(self.conv2_T)
        # print(self.conv2_S)

    def forward(self, x):
        # print(x.shape)
        # exit()
        xR = x[0]
        xT = x[1]
        # print(xR.shape)
        # print(xT.shape)
        # exit()
        residualR = xR
        residualT = xT
        if self.bn1_S is not None:
            outR = torch.cat([self.bn1_R(self.conv1_R(xR)), self.bn1_S(self.conv1_S(xR))], dim=1)
        else:
            outR = self.bn1_R(self.conv1_R(xR))
        # outR = self.bn1_R(outR)
        outR = self.relu(outR)

        if self.bn1_S is not None:
            outT = torch.cat([self.bn1_T(self.conv1_T(xT)), self.bn1_S(self.conv1_S(xT))], dim=1)
        # outT = self.bn1(outT)
        else:
            outT = self.bn1_T(self.conv1_T(xT))
        outT = self.relu(outT)
        # print(self.conv2_S(outR).shape)

        if self.bn2_S is not None:
            outR = torch.cat([self.bn2_R(self.conv2_R(outR)), self.bn2_S(self.conv2_S(outR))], dim=1)
        else:
            outR = self.bn2_R(self.conv2_R(outR))
        # outR = self.bn2(outR)

        if self.bn2_S is not None:
            outT = torch.cat([self.bn2_T(self.conv2_T(outT)), self.bn2_S(self.conv2_S(outT))], dim=1)
        # outT = self.bn2(outT)
        else:
            outT = self.bn2_T(self.conv2_T(outT))

        if self.downsample is not None:
            residualR, residualT = self.downsample([xR, xT])

        # print(residualR.shape)
        # print(outR.shape)

        outR += residualR
        outT += residualT
        outR = self.relu(outR)
        outT = self.relu(outT)

        return [outR, outT]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM, affine=affine)
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

    xR_O_self_NCC = torch.bmm(xR_O_N_C_WH, xR_O_N_WH_C)/(W*H)

    xT_O_N_WH_C = xT.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
    xT_O_N_C_WH = xT_O_N_WH_C.permute([0, 2, 1]).contiguous()
    xT_O_self_NCC = torch.bmm(xT_O_N_C_WH, xT_O_N_WH_C)/(W*H)

    # xR_O_self_NCC = (xR_O_self_NCC-xR_O_self_NCC.mean())/xR_O_self_NCC.std()
    # xT_O_self_NCC = (xT_O_self_NCC - xT_O_self_NCC.mean()) / xT_O_self_NCC.std()

    return xR_O_self_NCC, xT_O_self_NCC


class CoMatch(nn.Module):
    def __init__(self, channel):
        super(CoMatch, self).__init__()
        self.channnel = channel
        self.weight = torch.nn.Parameter(data=torch.rand((channel, channel), dtype=torch.float),
                                         requires_grad=True)

    def forward(self, xR, xT):
        assert xR.shape == xT.shape
        shape = xR.shape
        assert len(shape) == 4 and shape[1] == self.channnel
        N = shape[0]
        C = shape[1]
        W = shape[2]
        H = shape[3]

        xT_O_N_WH_C = xT.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        xT_O_N_C_WH = xT_O_N_WH_C.permute([0, 2, 1]).contiguous()
        xT_O_self_NCC = torch.bmm(xT_O_N_C_WH, xT_O_N_WH_C)/(W*H)

        xR_O_N_WH_C = xR.permute([0, 2, 3, 1]).contiguous().view([N, W * H, C])
        h = torch.bmm(xR_O_N_WH_C, self.weight.unsqueeze(0).expand((N, C, C)))/C

        # ret = h
        ret = torch.bmm(h, xT_O_self_NCC.detach())/C
        return ret.view(N, W, H, C).permute([0, 3, 1, 2]).contiguous()


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, share, thermal_weight=0.5):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False
        self.thermal_weight = thermal_weight

        super(PoseResNet, self).__init__()
        self.conv1_R = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1_T = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_R = nn.BatchNorm2d(64, momentum=BN_MOMENTUM, affine=affine)
        self.bn1_T = nn.BatchNorm2d(64, momentum=BN_MOMENTUM, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.comatch1 = CoMatch(64)

        self.layer1 = self._make_layer(block, 64, layers[0], share=share[0])
        self.comatch2 = CoMatch(64)
        # if attention:
        #     self.att_fusion1 = Att_fusion(64)
        self.layer2 = self._make_layer(block, 128, layers[1], share=share[1], stride=2)
        self.comatch3 = CoMatch(128)
        # if attention:
        #     self.att_fusion2 = Att_fusion(128)
        self.layer3 = self._make_layer(block, 256, layers[2], share=share[2], stride=2)
        self.comatch4 = CoMatch(256)
        # if attention:
        #     self.att_fusion3 = Att_fusion(256)
        self.layer4 = self._make_layer(block, 512, layers[3], share=share[3], stride=2)
        self.comatch5 = CoMatch(512)
        # if attention:
        #     self.att_fusion4 = Att_fusion_final(512)
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

    def _make_layer(self, block, planes, blocks, share=0.5, stride=1):
        # aa = block(self.inplanes, planes, stride)
        # print(aa)
        # print(stride)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = downsample_share(self.inplanes, planes, block.expansion, stride, ratio=share)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, ratio=share))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ratio=share))

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
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=affine))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        xR = x[:, 0:3, :, :]
        xT = x[:, 3:4, :, :].expand_as(xR)

        xR = self.conv1_R(xR)
        xR = self.bn1_R(xR)
        xR = self.relu(xR)
        xR = self.maxpool(xR)

        xT = self.conv1_T(xT)
        xT = self.bn1_T(xT)
        xT = self.relu(xT)
        xT = self.maxpool(xT)
        ret = {}
        ret['xR_G_att'] = []
        ret['xT_att'] = []
        ret['comatch_xR_G'] = []
        ret['comatch_xR_O'] = []
        # print(xR.max())
        # print(xT.max())

        ret['comatch_xR_O'] = ret['comatch_xR_O'] + [xR]
        xR_G = self.comatch1(xR, xT)
        # print(xR_G.max())
        # exit()
        ret['comatch_xR_G'] = ret['comatch_xR_G'] + [xR_G]

        R_G_att, T_att = get_att(xR_G, xT)
        ret['xR_G_att'] = ret['xR_G_att'] + [R_G_att]
        ret['xT_att'] = ret['xT_att'] + [T_att]

        xR, xT = self.layer1([xR_G, xT])
        ret['comatch_xR_O'] = ret['comatch_xR_O'] + [xR]
        xR_G = self.comatch2(xR, xT)
        ret['comatch_xR_G'] = ret['comatch_xR_G'] + [xR_G]

        R_G_att, T_att = get_att(xR_G, xT)
        ret['xR_G_att'] = ret['xR_G_att'] + [R_G_att]
        ret['xT_att'] = ret['xT_att'] + [T_att]



        # if attention:
        #     x = self.att_fusion1(xR, xT)

        xR, xT = self.layer2([xR_G, xT])
        ret['comatch_xR_O'] = ret['comatch_xR_O'] + [xR]
        xR_G = self.comatch3(xR, xT)
        ret['comatch_xR_G'] = ret['comatch_xR_G'] + [xR_G]

        R_G_att, T_att = get_att(xR_G, xT)
        ret['xR_G_att'] = ret['xR_G_att'] + [R_G_att]
        ret['xT_att'] = ret['xT_att'] + [T_att]
        # if attention:
        #     x = self.att_fusion2(xR, xT)

        xR, xT = self.layer3([xR_G, xT])
        ret['comatch_xR_O'] = ret['comatch_xR_O'] + [xR]
        xR_G = self.comatch4(xR, xT)
        ret['comatch_xR_G'] = ret['comatch_xR_G'] + [xR_G]

        R_G_att, T_att = get_att(xR_G, xT)
        ret['xR_G_att'] = ret['xR_G_att'] + [R_G_att]
        ret['xT_att'] = ret['xT_att'] + [T_att]
        # if attention:
        #     x = self.att_fusion3(xR, xT)

        xR, xT = self.layer4([xR_G, xT])
        ret['comatch_xR_O'] = ret['comatch_xR_O'] + [xR]
        xR_G = self.comatch5(xR, xT)
        ret['comatch_xR_G'] = ret['comatch_xR_G'] + [xR_G]

        R_G_att, T_att = get_att(xR_G, xT)
        ret['xR_G_att'] = ret['xR_G_att'] + [R_G_att]
        ret['xT_att'] = ret['xT_att'] + [T_att]

        if attention:
            x = self.att_fusion4(xR, xT)
        else:
            x = (1-self.thermal_weight)*xR + self.thermal_weight*xT
        # exit()
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


    def forward_T(self, x):
        xR = x[:, 0:3, :, :]
        xT = x[:, 3:4, :, :].expand_as(xR)
        xR = torch.zeros_like(xT)
        xR = self.conv1_R(xR)
        xR = self.bn1_R(xR)
        xR = self.relu(xR)
        xR = self.maxpool(xR)

        xT = self.conv1_T(xT)
        xT = self.bn1_T(xT)
        xT = self.relu(xT)
        xT = self.maxpool(xT)
        ret = {}
        ret['xR_G_att'] = []
        ret['xT_att'] = []
        ret['comatch_xR_G'] = []
        ret['comatch_xR_O'] = []
        xR, xT = self.layer1([xR, xT])
        xR, xT = self.layer2([xR, xT])
        xR, xT = self.layer3([xR, xT])
        xR, xT = self.layer4([xR, xT])
        x = xT
        # exit()
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

    def load_shared_stat_dict(self, input, strict):
        pretrained_keys = input.keys()
        self_keys = self.state_dict().keys()
        print(pretrained_keys)
        for key in self_keys:
            key_split = key.split('.')
            if key in pretrained_keys:
                continue
            elif key_split[0].startswith("conv"):
                num = key_split[0].replace("conv", '')[0]
                target_name ='conv{:s}.weight'.format(num)
                self.__getattr__(key_split[0]).load_state_dict({'weight': input[target_name]})
                # print(key)
                # exit()
            elif key_split[0].startswith("bn"):
                num = key_split[0].replace("bn", '')[0]
                target_name_weight = 'bn{:s}.weight'.format(num)
                target_name_bias = 'bn{:s}.weight'.format(num)
                self.__getattr__(key_split[0]).load_state_dict(
                    {'weight': input[target_name_weight],
                     'bias': input[target_name_bias]}, strict=False
                )

            elif key_split[0].startswith("layer") and key_split[2].startswith("conv"):
                layernum = key_split[0].replace("layer", "")
                seqnum = key_split[1]
                convnum = key_split[2].replace("conv", "")[0]
                # self.layer
                channel = self.state_dict()[key].shape[0]
                target_name = "layer{:s}.{:s}.conv{:s}.weight".format(layernum, seqnum, convnum)
                if key_split[2].endswith("_T") or key_split[2].endswith("_R"):
                    # print(self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]).weight.shape)
                    # print(input[target_name][0:channel, :, :, :].shape)
                    self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]).load_state_dict(
                        {"weight": input[target_name][0:channel, :, :, :].detach()}
                    )
                    # self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]).weight.set_(
                    #     input[target_name][0:channel, :, :, :].detach())
                elif key_split[2].endswith("_S"):
                    self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]).load_state_dict(
                        {"weight": input[target_name][-channel:, :, :, :].detach()}
                    )
                else:
                    print(key)
            elif key_split[0].startswith("layer") and key_split[2].startswith("bn"):
                layernum = key_split[0].replace("layer", "")
                seqnum = key_split[1]
                bnnum = key_split[2].replace("bn", "")[0]
                type = key_split[3]
                # print(type)
                # exit()
                if type == 'weight':
                    target_name = "layer{:s}.{:s}.bn{:s}.weight".format(layernum,
                                                                        seqnum, bnnum)
                    channel = self.state_dict()[key].shape[0]
                    if key_split[2].endswith("_R") or key_split[2].endswith("_T"):
                        # print(self.__getattr__("layer" + layernum)[int(seqnum)])
                        # exit()
                        res = self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(
                            key_split[2]
                        ).load_state_dict(
                            {"weight": input[target_name][0:channel].detach()}, strict=False
                        )
                    elif key_split[2].endswith("_S"):
                        res = self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(
                            key_split[2]
                        ).load_state_dict(
                            {"weight": input[target_name][-channel:].detach()}, strict=False
                        )
                    else:
                        print(key)
                elif type == 'bias':
                    target_name = "layer{:s}.{:s}.bn{:s}.bias".format(layernum,
                                                                        seqnum, bnnum)
                    channel = self.state_dict()[key].shape[0]
                    if key_split[2].endswith("_R") or key_split[2].endswith("_T"):
                        # print(self.__getattr__("layer" + layernum)[int(seqnum)])
                        # exit()
                        res = self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(
                            key_split[2]
                        ).load_state_dict(
                            {"bias": input[target_name][0:channel].detach()}, strict=False
                        )
                        # print(res)
                        # exit()
                    elif key_split[2].endswith("_S"):
                        res = self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(
                            key_split[2]
                        ).load_state_dict(
                            {"bias": input[target_name][-channel:].detach()}, strict=False
                        )
                        # print(res)
                        # exit()
                    else:
                        print(key)
                else:
                    print(key)
                # self.layer
                # channel = self.state_dict()[key].shape[0]
            elif key_split[0].startswith("layer") and key_split[2] == "downsample":
                if key_split[3].startswith("conv"):
                    layernum = key_split[0].replace("layer", "")
                    seqnum = key_split[1]
                    conv = key_split[3]
                    # self.layer
                    channel = self.state_dict()[key].shape[0]
                    target_name = "layer{:s}.{:s}.downsample.0.weight".format(layernum, seqnum)
                    if conv == "convR" or conv == "convT":
                        self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                            conv
                        ).load_state_dict(
                            {"weight": input[target_name][0:channel, :, :, :].detach()}
                            )
                        # print(input[target_name].shape)
                    elif conv == "convS":
                        self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                            conv
                        ).load_state_dict(
                            {"weight": input[target_name][-channel:, :, :, :].detach()}
                        )
                    else:
                        print(key)
                        # exit()
                        # self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]).load_state_dict(
                        #     {"weight": input[target_name][0:channel, :, :, :].detach()}
                        # )
                elif key_split[3].startswith("bn"):
                    layernum = key_split[0].replace("layer", "")
                    seqnum = key_split[1]
                    bn = key_split[3]
                    type = key_split[4]
                    # self.layer
                    if type == "weight":
                        target_name = "layer{:s}.{:s}.downsample.1.weight".format(layernum, seqnum)
                        channel = self.state_dict()[key].shape[0]
                        if bn == "bnR" or bn == "bnT":
                            res = self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                                bn
                            ).load_state_dict(
                                {"weight": input[target_name][0:channel].detach()}, strict=False
                            )
                            # print(res)
                        elif bn == "bnS":
                            res = self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                                bn
                            ).load_state_dict(
                                {"weight": input[target_name][-channel:].detach()}, strict=False
                            )
                            # print(res)
                        else:
                            print(key)
                    elif type == "bias":
                        target_name = "layer{:s}.{:s}.downsample.1.bias".format(layernum, seqnum)
                        # print(input[target_name].shape)
                        # print(self.state_dict()[key].shape)
                        # exit()
                        channel = self.state_dict()[key].shape[0]
                        if bn == "bnR" or bn == "bnT":
                            res = self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                                bn
                            ).load_state_dict(
                                {"bias": input[target_name][0:channel].detach()}, strict=False
                            )
                            # print(res)
                        elif bn == "bnS":
                            res = self.__getattr__("layer" + layernum)[int(seqnum)].downsample.__getattr__(
                                bn
                            ).load_state_dict(
                                {"bias": input[target_name][-channel:].detach()}, strict=False
                            )
                            # print(res)
                        else:
                            print(key)
                    # target_name = "layer{:s}.{:s}.downsample.0.weight".format(layernum, seqnum)

                else:
                    print(key)
            else:
                print(key)
        # print(input['layer2.0.downsample.1.bias'])
        # print(self.state_dict()['layer2.0.downsample.bnR.bias'])
        # print(self.state_dict()['layer2.0.downsample.bnT.bias'])
        # print(self.state_dict()['layer2.0.downsample.bnS.bias'])

                    # print(self.__getattr__("layer" + layernum)[int(seqnum)].__getattr__(key_split[2]))


                # print(channel)
                # print(input[target_name].shape)
                # print(self.__getattr__("layer"+layernum)[int(seqnum)])
                # print(key)


    def init_weights(self, num_layers):
        if 1:
            url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            pretrained_state_dict = torch.load('../weights/resnet18-5c106cde.pth')
            # print(pretrained_state_dict.keys())
            # print(self.state_dict().keys())
            self.load_shared_stat_dict(pretrained_state_dict, False)
            # exit()
            # print('=> loading pretrained model {}'.format(url))
            # self.load_state_dict(pretrained_state_dict, strict=False)
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock_share, [2, 2, 2, 2], [0.1, 0.2, 0.3, 0.4]),
               34: (BasicBlock_share, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256, adapt_thermal_weight=0.5, share_cut=0.0):
  block_class, layers, share = resnet_spec[num_layers]

  for i, sh in enumerate(share):
    share[i] = sh - share_cut
    if share[i]<0.0:
        share[i] = 0.0
  model = PoseResNet(block_class, layers, heads, head_conv=head_conv, share=share, thermal_weight=adapt_thermal_weight)
  model.init_weights(num_layers)
  return model
