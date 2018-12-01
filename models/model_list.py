from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import Atlas_ResNet
from models.bninception import Atlas_BNInception

def resnet18(pretrained=False, drop_rate=0.):
    return Atlas_ResNet(modeln="resnet18", pretrained=pretrained)

def resnet34(pretrained=False, drop_rate=0.):
    return Atlas_ResNet(modeln="resnet34", pretrained=pretrained)

def resnet50(pretrained=False, drop_rate=0.):
    return Atlas_ResNet(modeln="resnet50", pretrained=pretrained)

def resnet101(pretrained=False, drop_rate=0.):
    return Atlas_ResNet(modeln="resnet101", pretrained=pretrained)

def resnet152(pretrained=False, drop_rate=0.):
    return Atlas_ResNet(modeln="resnet152", pretrained=pretrained)


def densenet121(pretrained=False, drop_rate=0.):
    return Atlas_DenseNet(modeln="densenet121", pretrained=pretrained, 
            drop_rate=drop_rate)

def densenet169(pretrained=False, drop_rate=0.):
    return Atlas_DenseNet(modeln="densenet169", pretrained=pretrained, 
            drop_rate=drop_rate)

def densenet201(pretrained=False, drop_rate=0.):
    return Atlas_DenseNet(modeln="densenet201", pretrained=pretrained, 
            drop_rate=drop_rate)

def densenet161(pretrained=False, drop_rate=0.):
    return Atlas_DenseNet(modeln="densenet161", pretrained=pretrained, 
            drop_rate=drop_rate)

def bninception(pretrained=False, drop_rate=0.):
    return Atlas_BNInception(pretrained=pretrained, drop_rate=drop_rate)


def uselessnet():
    return UselessNet()