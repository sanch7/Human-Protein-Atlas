from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import Atlas_ResNet
from models.inception import Atlas_Inception
from models.xception import Atlas_Xception
from models.senet import Atlas_SENet
from models.sononet import Atlas_Sononet_Attn
from models.xception_attention import Atlas_Xception_Attn
from models.pnasnet import Atlas_PNASnet
from models.Jongchan.bamnet import Atlas_BAMNet
from models.dilatedrn import Atlas_DRN
from models.airx import Atlas_AirX

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

def inceptionv4(pretrained=False, drop_rate=0., num_channels=4):
    return Atlas_Inception(model_name = 'inceptionv4', pretrained=pretrained, 
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

def resnet18cbam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet18cbam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet18bam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet18bam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet34cbam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet34cbam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet34bam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet34bam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet50cbam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet50cbam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet50bam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet50bam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet101cbam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet101cbam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def resnet101bam(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_BAMNet(model_name = "resnet101bam", drop_rate=0., pretrained=False, 
                        num_channels=4)

def drnd54(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_DRN(model_name = "drn-d-54", drop_rate=0., pretrained=True, 
                        num_channels=4)

def drnd105(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_DRN(model_name = "drn-d-105", drop_rate=0., pretrained=True, 
                        num_channels=4)

def airx50(pretrained=False, drop_rate=0, num_channels=4):
    return Atlas_AirX(model_name = "airx50_32x4d", drop_rate=0., pretrained=True, 
                        num_channels=4)

def uselessnet():
    return UselessNet()