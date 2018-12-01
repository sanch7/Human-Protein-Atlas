from torchvision import models
from pretrainedmodels.models import bninception
from torch import nn
import torch.nn.functional as F

def Atlas_BNInception(pretrained=False, drop_rate=0.):
    print("Using BN Inception")
    if pretrained:
        model = bninception(pretrained="imagenet")
    else:
        model = bninception(pretrained=None)
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    
    nconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))

    if pretrained:
        print('Loading weights...')
        nconv.weight.data[:,:3,:,:] = model.conv1_7x7_s2.weight.data.clone()
        nconv.weight.data[:,3,:,:] = model.conv1_7x7_s2.weight.data[:,1,:,:].clone()
    
    model.conv1_7x7_s2 = nconv
    
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 28),
            )

    return model