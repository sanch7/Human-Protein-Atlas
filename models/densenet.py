import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import densenet121, densenet169, densenet201, densenet161

def Atlas_DenseNet(modeln = "densenet121", drop_rate=0., pretrained=False):
    """
    Params:
        model: ['densenet121', 'densenet169', 'densenet201', 'densenet161']
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """
    if modeln == "densenet121":
        model = densenet121
        cin_features = 1024
    elif modeln == "densenet169":
        model = densenet169
        cin_features = 1664
    elif modeln == "densenet201":
        model = densenet201
        cin_features = 1920
    elif modeln == "densenet161":
        model = densenet161
        cin_features = 2208
    else:
        raise ValueError('Model name not recognized.')
    model = model(pretrained=pretrained, drop_rate=drop_rate)

    model.classifier = nn.Linear(cin_features*4, 28)

    if modeln != "densenet161":
        nconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained:
            print('Loading weights...')
            nconv.weight.data[:,:3,:,:] = model.features.conv0.weight.data.clone()
            nconv.weight.data[:,3,:,:] = model.features.conv0.weight.data[:,1,:,:].clone()

    else:
        nconv = nn.Conv2d(4, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained:
            print('Loading weights...')
            nconv.weight.data[:,:3,:,:] = model.features.conv0.weight.data.clone()
            nconv.weight.data[:,3,:,:] = model.features.conv0.weight.data[:,1,:,:].clone()
    
    model.features.conv0 = nconv

    return model

