from torchvision import models
from pretrainedmodels.models import bninception, inceptionv4, inceptionresnetv2
from torch import nn
import torch.nn.functional as F

def Atlas_Inception(model_name, pretrained=False, drop_rate=0., num_channels=4):
    if model_name in ['bninception', 'inceptionv2']:
        print("Using BN Inception")
        if pretrained:
            print('Loading weights...')
            model = bninception(pretrained="imagenet")
        else:
            model = bninception(pretrained=None)
        model.global_pool = nn.AdaptiveAvgPool2d(1)
        
        if num_channels not in [3, 4]:
            raise ValueError('num_channels should be 3 or 4.')

        if num_channels == 4:
            nconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

            if pretrained:
                nconv.weight.data[:,:3,:,:] = model.conv1_7x7_s2.weight.data.clone()
                nconv.weight.data[:,3,:,:] = model.conv1_7x7_s2.weight.data[:,1,:,:].clone()
            
            model.conv1_7x7_s2 = nconv
        
        model.last_linear = nn.Sequential(
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 28),
                )

    if model_name == 'inceptionresnetv2':
        print("Using Inception Resnet v2")
        if pretrained:
            print('Loading weights...')
            model = inceptionresnetv2(pretrained="imagenet")
        else:
            model = inceptionresnetv2(pretrained=None)
        model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        
        if num_channels not in [3, 4]:
            raise ValueError('num_channels should be 3 or 4.')

        if num_channels == 4:
            nconv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

            if pretrained:
                nconv.weight.data[:,:3,:,:] = model.conv2d_1a.conv.weight.data.clone()
                nconv.weight.data[:,3,:,:] = model.conv2d_1a.conv.weight.data[:,1,:,:].clone()
            
            model.conv2d_1a.conv = nconv
        
        model.last_linear = nn.Sequential(
                    nn.BatchNorm1d(1536),
                    nn.Dropout(0.5),
                    nn.Linear(1536, 28),
                )

    return model