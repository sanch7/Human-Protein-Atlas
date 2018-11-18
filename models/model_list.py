from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import ResNet

def resnet():
    return ResNet()

def densenet121(pretrained=False, drop_rate=0.):
    return Atlas_DenseNet(modeln="densenet121", pretrained=pretrained, 
    		drop_rate=drop_rate)

def uselessnet():
    return UselessNet()