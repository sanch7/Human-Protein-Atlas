from torchvision import models
from pretrainedmodels.models import xception
from .xceptionalt import xceptionalt
from torch import nn
import torch.nn.functional as F

def Atlas_Xception(model_name = "xception", pretrained=False, drop_rate=0., 
                        num_channels=4):
    if model_name == "xception":
        print("Using Xception")
        if pretrained:
            print('Loading weights...')
            model = xception(pretrained="imagenet")
        else:
            model = xception(pretrained=None)
        
        if num_channels not in [3, 4]:
            raise ValueError('num_channels should be 3 or 4.')

        if num_channels == 4:
            nconv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

            if pretrained:
                nconv.weight.data[:,:3,:,:] = model.conv1.weight.data.clone()
                nconv.weight.data[:,3,:,:] = model.conv1.weight.data[:,1,:,:].clone()
            
            model.conv1 = nconv
        
        model.last_linear = nn.Linear(in_features=2048, out_features=28, bias=True)

    if model_name == "xceptionalt":
        print("Using XceptionAlt")
        if pretrained:
            print('Loading weights...')
            model = xceptionalt(pretrained=True)
        else:
            model = xceptionalt(pretrained=False)
        
        if num_channels not in [3, 4]:
            raise ValueError('num_channels should be 3 or 4.')

        if num_channels == 4:
            nconv = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

            if pretrained:
                nconv.weight.data[:,:3,:,:] = model.conv1.weight.data.clone()
                nconv.weight.data[:,3,:,:] = model.conv1.weight.data[:,1,:,:].clone()
            
            model.conv1 = nconv
        
        model.fc = nn.Linear(in_features=2048, out_features=28, bias=True)

    return model