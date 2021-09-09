""" This code is only for educational purposes.

---
ResNet(B) implementation https://arxiv.org/pdf/1603.05027.pdf with preactivation and optional bottleneck. 

(B) The projection shortcut is used to match dimensions (done by 1×1 convolutions). https://arxiv.org/pdf/1512.03385.pdf


Code inspiration: 
https://github.com/eemlcommunity/PracticalSessions2020/blob/master/sup_tutorial/resnet18_cifar10_baseline_solution.ipynb
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
---

"""

import torch.nn as nn
import torch
from typing import Sequence


class BlockV2(nn.Module):

    def __init__(self, in_channels: int, channels: int, bottleneck: bool, stride: int = 1):
        super(BlockV2, self).__init__()

        # define preactivation
        self.pre_activation = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU())

        # define residual function without preactivation
        channel_div = 4 if bottleneck else 1
        residual_funtion = [
            nn.Conv2d(in_channels=in_channels,
                      out_channels=channels // channel_div,
                      kernel_size=1 if bottleneck else 3,
                      stride=1 if bottleneck else stride,
                      padding=0 if bottleneck else 1,
                      bias=False),

            nn.BatchNorm2d(num_features=channels // channel_div),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // channel_div,
                      out_channels=channels // channel_div,
                      kernel_size=3,
                      stride=stride if bottleneck else 1,
                      padding=1,
                      bias=False)]

        if bottleneck:
            residual_funtion.append(nn.BatchNorm2d(
                num_features=channels // channel_div))
            residual_funtion.append(nn.ReLU())
            residual_funtion.append(nn.Conv2d(in_channels=channels // channel_div,
                                              out_channels=channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=False))

        self.residual_funtion = nn.Sequential(*residual_funtion)

        # define schorcut
        self.use_projection = stride != 1 or in_channels != channels
        if self.use_projection:
            self.shortcut_function = nn.Conv2d(in_channels=in_channels,
                                               out_channels=channels,
                                               kernel_size=1,
                                               stride=stride,
                                               bias=False)

    def forward(self, x):
        shorcut = x

        x = self.pre_activation(x)

        if self.use_projection:
            shorcut = self.shortcut_function(x)

        return self.residual_funtion(x) + shorcut


class BlockGroup(nn.Module):

    def __init__(self, num_blocks: int, in_channels: int, channels: int, bottleneck: bool, stride: int):
        super(BlockGroup, self).__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                BlockV2(in_channels=in_channels if i == 0 else channels,
                        channels=channels,
                        bottleneck=bottleneck,
                        stride=stride if i == 0 else 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):

    def __init__(self, image_channels: int,
                 num_blocks_per_group: Sequence[int],
                 num_classes: int,
                 bottleneck: bool,
                 channels_per_group: Sequence[int]):
        super(ResNet, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=image_channels,
                               out_channels=channels_per_group[0],
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)

        groups = []
        stride_per_group = (1, 2, 2, 2)
        for i, num_blokcs in enumerate(num_blocks_per_group):
            groups.append(BlockGroup(num_blocks=num_blokcs,
                          in_channels=channels_per_group[i],
                          channels=channels_per_group[i+1 if i +
                                                      1 < len(channels_per_group) else i],
                          bottleneck=bottleneck,
                          stride=stride_per_group[i]))
        self.groups = nn.Sequential(*groups)

        self.pre_activation = nn.Sequential(
            nn.BatchNorm2d(num_features=channels_per_group[-1]),
            nn.ReLU())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.maxpool(x)
        x = self.groups(x)
        x = self.pre_activation(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def resnet18(image_channels: int = 3, num_classes: int = 10, bottleneck: bool = False, pretrained: bool = False) -> ResNet:
    model = ResNet(image_channels=image_channels,
                   num_blocks_per_group=(2, 2, 2, 2),
                   num_classes=num_classes,
                   bottleneck=bottleneck,
                   channels_per_group=(64, 128, 256, 512))

    if pretrained:
        _path = "vision/ResNet/pretrain_models/"
        path = _path + "pretrained_resnet18_bottleneck.pth" if bottleneck else _path + \
            "pretrained_resnet18.pth"

        model.load_state_dict(torch.load(path))
        model.eval()
    return model
