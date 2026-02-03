import torch.nn as nn
from torchvision import models


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.backbone(x)
