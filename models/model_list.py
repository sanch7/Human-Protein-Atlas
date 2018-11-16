from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import ResNet

def resnet():
    return ResNet()

def densenet121():
    return Atlas_DenseNet()

def uselessnet():
    return UselessNet()