import torch
import torch.nn as nn
from torchvision import transforms
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, 
    Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate, Normalize,
    Resize
)


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

def alb_transform_train(imsize = 256, p=1):
    albumentations_transform = Compose([
    Resize(imsize, imsize), 
    RandomRotate90(),
    Flip(),
    Transpose(),
    OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
    OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
    OneOf([
        OpticalDistortion(p=0.3),
        GridDistortion(p=.1),
        IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
    OneOf([
        # CLAHE(clip_limit=2),
        IAASharpen(),
        IAAEmboss(),
        RandomContrast(),
        RandomBrightness(),
        ], p=0.3),
    Normalize(
        mean=[0.485, 0.456, 0.406, 0.456],
        std=[0.229, 0.224, 0.225, 0.224]
        )
    ], p=p)
    return albumentations_transform

def alb_transform_test(imsize = 256, p=1):
    albumentations_transform = Compose([
    Resize(imsize, imsize), 
    RandomRotate90(),
    Flip(),
    Normalize(
        mean=[0.485, 0.456, 0.406, 0.456],
        std=[0.229, 0.224, 0.225, 0.224]
        )
    ], p=p)
    return albumentations_transform