from torch import nn
from torchvision.models.inception import Inception3

def Atlas_SENet(model_name = "seinceptionv3", pretrained=False, drop_rate=0., 
                        num_channels=4):
    if model_name == "seinceptionv3":
        print("Using SEInception3")
        model = SEInception3(28, aux_logits=False, num_channels=num_channels)

    return model

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEInception3(nn.Module):
    def __init__(self, num_classes, aux_logits=True, transform_input=False,
                    num_channels=4):
        super(SEInception3, self).__init__()
        model = Inception3(num_classes=num_classes, aux_logits=aux_logits,
                           transform_input=transform_input)
        model.Mixed_5b.add_module("SELayer", SELayer(192))
        model.Mixed_5c.add_module("SELayer", SELayer(256))
        model.Mixed_5d.add_module("SELayer", SELayer(288))
        model.Mixed_6a.add_module("SELayer", SELayer(288))
        model.Mixed_6b.add_module("SELayer", SELayer(768))
        model.Mixed_6c.add_module("SELayer", SELayer(768))
        model.Mixed_6d.add_module("SELayer", SELayer(768))
        model.Mixed_6e.add_module("SELayer", SELayer(768))
        if aux_logits:
            model.AuxLogits.add_module("SELayer", SELayer(768))
        model.Mixed_7a.add_module("SELayer", SELayer(768))
        model.Mixed_7b.add_module("SELayer", SELayer(1280))
        model.Mixed_7c.add_module("SELayer", SELayer(2048))

        if num_channels==4:
            model.Conv2d_1a_3x3.conv = nn.Conv2d(4, 32, kernel_size=(3, 3), 
                                                stride=(2, 2), bias=False)

        self.model = model

    def forward(self, x):
        _, _, h, w = x.size()
        # if (h, w) != (299, 299):
        #     raise ValueError("input size must be (299, 299)")

        return self.model(x)


def se_inception_v3(**kwargs):
    return SEInception3(**kwargs)
