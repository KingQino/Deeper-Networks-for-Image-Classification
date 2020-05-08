# -*- coding: utf-8 -*-
# @Time    : 2020/4/30 1:07 PM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : modified_model.py
# @Software: PyCharm
# Reference:
#   https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_learning
#   https://github.com/39239580/googlenet-pytorch/blob/master/Inception_v1_mnist.py
#   https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
import torch.nn as nn
import torch
import torch.nn.functional as F


class GoogLeNet_MNIST(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet_MNIST, self).__init__()

        self.conv1 = BasicConv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = BasicConv2d(8, 32, kernel_size=3, padding=1)

        self.inception3a = Inception(32, 16, 24, 32, 4, 8, 8)
        self.inception3b = Inception(64, 32, 32, 48, 8, 24, 16)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(120, 48, 24, 52, 4, 12, 16)
        self.inception4b = Inception(128, 40, 28, 56, 6, 16, 16)
        self.inception4c = Inception(128, 32, 32, 64, 12, 16, 16)
        self.inception4d = Inception(128, 28, 36, 72, 8, 16, 16)
        self.inception4e = Inception(132, 64, 40, 80, 8, 32, 32)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(208, 64, 40, 80, 8, 32, 32)
        self.inception5b = Inception(208, 96, 48, 96, 12, 32, 32)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 1 x 28 x 28
        x = self.conv1(x)
        # N x 8 x 28 x 28
        x = self.conv2(x)

        # N x 32 x 28 x 28
        x = self.inception3a(x)
        # N x 64 x 28 x 28
        x = self.inception3b(x)
        # N x 120 x 28 x 28
        x = self.maxpool1(x)
        # N x 120 x 14 x 14

        x = self.inception4a(x)
        # N x 128 x 14 x 14
        x = self.inception4b(x)
        # N x 128 x 14 x 14
        x = self.inception4c(x)
        # N x 128 x 14 x 14
        x = self.inception4d(x)
        # N x 132 x 14 x 14
        x = self.inception4e(x)
        # N x 208 x 14 x 14
        x = self.maxpool2(x)

        # N x 208 x 7 x 7
        x = self.inception5a(x)
        # N x 208 x 7 x 7
        x = self.inception5b(x)
        # N x 256 x 7 x 7

        x = self.avgpool(x)
        # N x 256 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 256
        x = self.dropout(x)
        x = self.fc(x)
        # N x 10 (num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # ensure the output size equals the input size
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # ensure the output size equals the input size
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)  # concatenate the branches in channel dimension | [batch, channel, high, width]


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)  # 2048 = 128 * 4 * 4
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)  # In the original paper, the dropout is 70%, but we set it as 50%
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x