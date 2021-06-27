from torchvision import models
import torch.nn as nn


class StableNet(nn.Module):

    def __init__(self, n_classes, pretrained=True):
        super().__init__()

        self.n_classes = n_classes

        res = models.resnet18(pretrained=pretrained)
        modules = [*res.children()]
        self.backbone = nn.Sequential(*modules[:-2])
        self.fc = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                nn.Flatten(),
                                nn.Linear(512, 512))
        self.fe = nn.Sequential(self.backbone, self.fc)
        self.classifier = nn.Linear(512, self.n_classes)

    def forward(self, x, need_feat=False):
        z = self.fe(x)
        y = self.classifier(z)
        if need_feat:
            return z, y
        return y
