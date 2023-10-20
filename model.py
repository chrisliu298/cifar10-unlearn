import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18(num_classes=10):
    net = resnet18(weights=None, num_classes=num_classes)
    net.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    return net


def get_cnn(num_classes=10):
    c = 64
    net = nn.Sequential(
        # Layer 1
        nn.Conv2d(3, c, 3, 1, 1, bias=False),
        nn.BatchNorm2d(c),
        nn.ReLU(),
        # Layer 2
        nn.Conv2d(c, 2 * c, 3, 1, 1, bias=False),
        nn.BatchNorm2d(2 * c),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(2 * c, 4 * c, 3, 1, 1, bias=False),
        nn.BatchNorm2d(4 * c),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 4
        nn.Conv2d(4 * c, 8 * c, 3, 1, 1, bias=False),
        nn.BatchNorm2d(8 * c),
        nn.ReLU(),
        nn.MaxPool2d(8),
        # Layer 5
        nn.Flatten(),
        nn.Linear(8 * c, num_classes),
    )
    return net
