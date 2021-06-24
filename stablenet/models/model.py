from torchvision import models
import torch.nn as nn


class StableNet(nn.Module):

    def __init__(self, n_classes, pretrained=True):
        super().__init__()

        self.n_classes = n_classes

        self.fe = models.resnet18(pretrained=pretrained)
        self.fe.fc = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, self.n_classes)

    def forward(self, x, need_feat=False):
        z = self.fe(x)
        y = self.classifier(z)
        if need_feat:
            return z, y
        return y
