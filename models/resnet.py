import torch
import torch.nn as nn 
from torchvision.models import resnet50

def ResNet():
    net = resnet50(pretrained=False)
    for p in net.parameters():
        p.requires_grad = True
    inft = net.fc.in_features
    net.fc = nn.Linear(in_features=inft, out_features=28)
    net.avgpool = nn.AdaptiveAvgPool2d(1)
    net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
    return net  
