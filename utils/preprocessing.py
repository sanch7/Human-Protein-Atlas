import torch
import torch.nn as nn
from torchvision import transforms

# https://pytorch.org/docs/master/torchvision/models.html
# Copied green channel values for the yellow channel
torchvision_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.456],
                                            std=[0.229, 0.224, 0.225, 0.224,])

def train_transformer(imsize = 256):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize(imsize),
        transforms.RandomRotation(40.0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        torchvision_normalize
        ]) 
    return train_tf

def test_transformer(imsize = 256):
    test_tf = transforms.Compose([
        transforms.ToPILImage(),
        # resize the image to 64x64 (remove if images are already 64x64)
        transforms.Resize(imsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        torchvision_normalize
        ]) 
    return test_tf