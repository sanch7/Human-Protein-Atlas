import torch
import torch.nn as nn 

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# def ResNet():
#     net = resnet50(pretrained=False)
#     for p in net.parameters():
#         p.requires_grad = True
#     inft = net.fc.in_features
#     net.fc = nn.Linear(in_features=inft, out_features=28)
#     net.avgpool = nn.AdaptiveAvgPool2d(1)
#     net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#     return net  

def Atlas_ResNet(modeln = "resnet34", pretrained=False):
    """
    Params:
        model: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    print("Using ResNet")
    if modeln == "resnet18":
        model = resnet18
        cin_features = 512
    elif modeln == "resnet34":
        model = resnet34
        cin_features = 512
    elif modeln == "resnet50":
        model = resnet50
        cin_features = 2048
    elif modeln == "resnet101":
        model = resnet101
        cin_features = 2048
    elif modeln == "resnet152":
        model = resnet152
        cin_features = 2048
    else:
        raise ValueError('Model name not recognized.')
    model = model(pretrained=pretrained)

    model.fc = nn.Linear(cin_features*4*25, 28)

    nconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if pretrained:
        print('Loading weights...')
        nconv.weight.data[:,:3,:,:] = model.conv1.weight.data.clone()
        nconv.weight.data[:,3,:,:] = model.conv1.weight.data[:,1,:,:].clone()
    
    model.conv1 = nconv

    return model