from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import Atlas_ResNet
from models.inception import Atlas_Inception
from models.xception import Atlas_Xception
from models.senet import Atlas_SENet
from models.sononet import Atlas_Sononet_Attn
from models.xception_attention import Atlas_Xception_Attn
from models.pnasnet import Atlas_PNASnet

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


def densenet121(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_DenseNet(modeln="densenet121", pretrained=pretrained, 
            drop_rate=drop_rate, num_channels=num_channels)

def densenet169(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_DenseNet(modeln="densenet169", pretrained=pretrained, 
            drop_rate=drop_rate, num_channels=num_channels)

def densenet201(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_DenseNet(modeln="densenet201", pretrained=pretrained, 
            drop_rate=drop_rate, num_channels=num_channels)

def densenet161(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_DenseNet(modeln="densenet161", pretrained=pretrained, 
            drop_rate=drop_rate, num_channels=num_channels)

def bninception(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Inception(model_name = 'bninception', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def inceptionv2(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Inception(model_name = 'bninception', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def inceptionresnetv2(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Inception(model_name = 'inceptionresnetv2', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def xception(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Xception(model_name = 'xception', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def xceptionalt(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Xception(model_name = 'xceptionalt', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def seinceptionv3(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_SENet(model_name = 'seinceptionv3', pretrained=pretrained, 
                                drop_rate=drop_rate, num_channels=num_channels)

def sononet_grid_attention(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Sononet_Attn(model_name = "sononet_grid_attention", pretrained=False, 
                                drop_rate=0., num_channels=4)

def xception_grid_attention(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Xception_Attn(model_name = "xception_grid_attention", pretrained=False, 
                                drop_rate=0., num_channels=4)

def pnasnet(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_PNASnet(model_name = "pnasnet", pretrained=False, drop_rate=0., 
                        num_channels=4)

def uselessnet():
    return UselessNet()