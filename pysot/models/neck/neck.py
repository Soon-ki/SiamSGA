# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch
from pysot.core.config import cfg


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2), AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)

        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out


class AdjustChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.channel_reduce1 = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], 1),
            nn.BatchNorm2d(out_channels[0], eps=0.001),
        )
        self.channel_reduce2 = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], 1),
            nn.BatchNorm2d(out_channels[1], eps=0.001),
        )
        self.channel_reduce3 = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], 1),
            nn.BatchNorm2d(out_channels[2], eps=0.001),
        )

    def forward(self, features):

        f1 = self.channel_reduce1(features[0])

        f2 = self.channel_reduce2(features[1])
        f3 = self.channel_reduce3(features[2])

        return f1, f2, f3