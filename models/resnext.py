import torch
import torchvision
from config.value_config import *

def make_model(key):
    return ResNext(key)


class ResNext(torch.nn.Module):
    def __init__(self, key):
        super(ResNext, self).__init__()
        backbone = torchvision.models.__dict__[key](pretrained=True)
        self.layer0 = torch.nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=self.layer4[-1].conv1.in_channels, out_features=NUMCLASS),
        )

        pass

    def forward(self, x, weights=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


