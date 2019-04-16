#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   simple_cnn.py
@Time    :   2019/04/14 22:21:39
@Author  :   Wu 
@Version :   1.0
@Desc    :   None
'''


import torch
import torch.nn as nn

# import the files of mine
from logger import log


class MyCNN(nn.Module):
    def __init__(self, image_size, num_class, img_in_channels):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=img_in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # fully connected layer
        self.fc = nn.Linear(32 * (image_size // 4) * (image_size // 4), num_class)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size\\
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output