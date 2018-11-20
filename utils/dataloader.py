from __future__ import print_function, division
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split

from skimage import io, transform
import cv2
from .preprocessing import alb_transform_train, alb_transform_test

import warnings
warnings.filterwarnings("ignore")

train_labels_path = f"./data/train.csv"
test_submission_path = f"./data/sample_submission.csv"
train_images_path = f"./data/train/"
test_images_path = f"./data/test/"

labels_dict={ 0: "Nucleoplasm", 1: "Nuclear membrane", 2: "Nucleoli", 
    3: "Nucleoli fibrillar center", 4: "Nuclear speckles", 5: "Nuclear bodies", 
    6: "Endoplasmic reticulum", 7: "Golgi apparatus", 8: "Peroxisomes", 
    9: "Endosomes", 10: "Lysosomes", 11: "Intermediate filaments", 
    12: "Actin filaments", 13: "Focal adhesion sites", 14: "Microtubules", 
    15: "Microtubule ends", 16: "Cytokinetic bridge", 17: "Mitotic spindle", 
    18: "Microtubule organizing center", 19: "Centrosome", 20: "Lipid droplets", 
    21: "Plasma membrane", 22: "Cell junctions", 23: "Mitochondria", 
    24: "Aggresome", 25: "Cytosol", 26: "Cytoplasmic bodies", 27: "Rods & rings"}

color_channels = ('red','green','blue','yellow')

def label_gen(labelstr):
    label = torch.zeros(28)
    labelstr = labelstr.split()
    for l in labelstr:
        label[int(l)]=1
    return label

class ProteinDataset(Dataset):
    def __init__(self, data_df = None, test = False, imsize = 256, 
                    transformer = None, preload=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            test: load testing images
            imsize: output image size
        """
        super(ProteinDataset, self).__init__()
        self.test = test
        self.imsize = imsize
        self.transformer = transformer
        self.preload = preload
        self.images_path = test_images_path if test else train_images_path
        if data_df is None:
            self.images_df = pd.read_csv(train_labels_path)
        else:
            if 'Target' not in data_df.columns:
                data_df['Target'] = torch.zeros(len(data_df), 28)
            self.images_df = data_df

        if not self.test:
            self.images_df['Target'] = self.images_df['Target'].apply(label_gen)

        if preload:
            print('Preloading images...')
            self.imarray = np.zeros((len(self.images_df), self.imsize, 
                                        self.imsize, 4), dtype='uint8')
            for idx, imagename in enumerate(tqdm(self.images_df['Id'])):
                for ch, channel in enumerate(color_channels):
                    imagepath = self.images_path + imagename + '_' + channel + ".png"
                    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.imsize, self.imsize), 
                                        interpolation=cv2.INTER_AREA)
                    self.imarray[idx,:,:,ch] = img

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Id']
        if self.preload:
            image = self.imarray[idx,:,:,:]
        else:
            image = np.zeros((512, 512, 4), dtype='uint8')
            for ch, channel in enumerate(color_channels):
                imagepath = self.images_path + imagename + '_' + channel + ".png"
                img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)     #232s
                # img = io.imread(imagepath)                            #239s
                # img = Image.open(imagepath)                           #236s
                image[:,:, ch] = img

        if self.transformer:
            image = self.transformer(image=image)['image']
        else:
            image = transform.resize(image, (self.imsize, self.imsize))
        image = torch.from_numpy(image).permute(-1, 0, 1).float()
        targets = self.images_df['Target'][idx]    
        return image, targets, imagename

    def getImageName(self, imagename):
        image = np.zeros((512, 512, 4), dtype='uint8')
        for ch, channel in enumerate(color_channels):
            imagepath = self.images_path + imagename + '_' + channel + ".png"
            img = Image.open(imagepath)
            image[:,:, ch] = img

        targets = self.images_df[self.images_df['Id'] == imagename]['Target']    
        return image, targets
        
def get_data_loaders(imsize=256, batch_size=16, test_size=0.15, num_workers=4, 
                        preload=False, eval_mode=False):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(train_labels_path)
    train_df, valid_df = train_test_split(images_df, test_size=test_size, random_state=42)
    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()

    # set up the transformers
    if eval_mode:
        train_tf = alb_transform_test(imsize)
    else:
        train_tf = alb_transform_train(imsize)
    valid_tf = alb_transform_test(imsize)
    # train_tf = train_transformer(imsize)
    # valid_tf = test_transformer(imsize)

    # set up the datasets
    train_dataset = ProteinDataset(data_df = train_df, imsize=imsize, 
                                    transformer = train_tf, preload=preload)
    valid_dataset = ProteinDataset(data_df = valid_df, imsize=imsize, 
                                    transformer = valid_tf, preload=preload)

    train_sampler = SubsetRandomSampler(train_df.index)
    valid_sampler = SubsetRandomSampler(valid_df.index) 

    # set up the data loaders
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   sampler=train_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   sampler=valid_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return train_loader, valid_loader

def get_test_loader(imsize=256, batch_size=16, num_workers=4):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(test_submission_path)

    # set up the transformer
    test_tf = alb_transform_test(imsize)

    # set up the datasets
    test_dataset = ProteinDataset(data_df = images_df, imsize=imsize, 
                                    transformer = test_tf, test=True)

    # set up the data loaders
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return test_loader