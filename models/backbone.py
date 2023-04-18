import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Resnet101FPN(nn.Module):
    def __init__(self, mode):
        super().__init__()
        assert mode in ['TRAIN', 'EVAL'], 'Wrong mode!'
        self.mode = mode
        # get resnet101
        if self.mode == 'TRAIN':
            self.resnet101 = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        else:
            self.resnet101 = torchvision.models.resnet101(weights=None)
        self.resnet101.avgpool = None
        self.resnet101.fc = None
        # base layers
        self.base_layer0 = nn.Sequential(self.resnet101.conv1, self.resnet101.bn1, self.resnet101.relu, self.resnet101.maxpool)
        self.base_layer1 = self.resnet101.layer1
        self.base_layer2 = self.resnet101.layer2
        self.base_layer3 = self.resnet101.layer3
        self.base_layer4 = self.resnet101.layer4
        # lateral layers
        self.lateral_layer0 = nn.Conv2d(2048, 256, 1, 1, 0)
        self.lateral_layer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.lateral_layer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.lateral_layer3 = nn.Conv2d(256, 256, 1, 1, 0)
        # smooth layers
        self.smooth_layer0 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth_layer1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth_layer2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth_layer3 = nn.Conv2d(256, 256, 3, 1, 1)
        # down-sample
        self.down_sample_layer = nn.MaxPool2d(2, 2, 0)
        # initialize
        if self.mode == 'TRAIN':
            self.initialize()

    def forward(self, x):
        # bottom-up
        x = self.base_layer0(x)
        c2 = self.base_layer1(x)
        c3 = self.base_layer2(c2)
        c4 = self.base_layer3(c3)
        c5 = self.base_layer4(c4)
        # top-down
        # 此处的c实际为p，因为要节省内存
        c5 = self.lateral_layer0(c5)
        c4 = self.merge(c5, self.lateral_layer1(c4))
        c3 = self.merge(c4, self.lateral_layer2(c3))
        c2 = self.merge(c3, self.lateral_layer3(c2))
        # smooth
        c2 = self.smooth_layer0(c2)
        c3 = self.smooth_layer1(c3)
        c4 = self.smooth_layer2(c4)
        c5 = self.smooth_layer3(c5)
        c6 = self.down_sample_layer(c5)
        return [c2, c3, c4, c5, c6]

    @staticmethod
    def merge(p, c):
        _, _, h, w = c.size()
        return F.interpolate(p, [h, w]) + c

    def initialize(self):
        for layer in [self.lateral_layer0, self.lateral_layer1, self.lateral_layer2, self.lateral_layer3,
                      self.smooth_layer0, self.smooth_layer1, self.smooth_layer2, self.smooth_layer3]:
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias != None:
                nn.init.constant_(layer.bias, 0)
