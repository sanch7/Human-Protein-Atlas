"""
Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)

@author: tstandley
Adapted by cadene

Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

# Attention
from .sononet import GridAttentionBlock2D_TORR

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception_Attn(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, in_channels=3, nonlocal_mode='concatenation', aggregation_mode='concat'):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception_Attn, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        # self.fc = nn.Linear(2048, num_classes)

        filters = [728, 728, 1536, 2048];
        ################
        # Attention Maps
        self.compatibility_score1 = GridAttentionBlock2D_TORR(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = GridAttentionBlock2D_TORR(in_channels=filters[3], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(filters[2]+filters[3]+filters[3], num_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier1 = nn.Linear(filters[2], num_classes)
            self.classifier2 = nn.Linear(filters[3], num_classes)
            self.classifier3 = nn.Linear(filters[3], num_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[2] + filters[3] + filters[3], num_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(num_classes*3, num_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        ####################

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        preds =  self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))

    # def logits(self, features):
    #     x = self.relu(features)

    #     x = F.adaptive_avg_pool2d(x, (1, 1))
    #     x = x.view(x.size(0), -1)
    #     x = self.last_linear(x)
    #     return x

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        block5 = self.block5(x)
        x = self.block6(block5)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        conv3 = self.conv3(x)
        x = self.bn3(conv3)
        x = self.relu(x)

        conv4 = self.conv4(x)
        # x = self.bn4(x)

        # x = self.logits(x)
        batch_size = input.shape[0]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        pooled = F.adaptive_avg_pool2d(conv4, (1, 1)).view(batch_size, -1)

        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(block5, conv4)
        g_conv2, att2 = self.compatibility_score2(conv3, conv4)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)

        return self.aggregate(g1, g2, pooled)


def xception(num_classes=1000, pretrained='imagenet', in_channels=3):
    model = Xception_Attn(num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception_Attn(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    # model.last_linear = model.fc
    # del model.fc
    return model
