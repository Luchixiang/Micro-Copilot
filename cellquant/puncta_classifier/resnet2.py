import torch
from torchvision.models import resnet50
import torch.nn as nn
import torchvision.models


class PunctaResnet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = resnet18(pretrained=True)
        # self.net = resnet34(pretrained=True)
        self.net = resnet50(weights=None)
        self.net.conv1 = torch.nn.Conv2d(6, 64, (7, 7), (2, 2), (3, 3), bias=False)
        n_inputs = self.net.fc.in_features
        self.net.fc = nn.Linear(n_inputs, 3)
    def forward(self, x):
        return self.net(x)



