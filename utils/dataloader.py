from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import utils, transforms
from sklearn.model_selection import train_test_split

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

class ProteinDataset(Dataset):
    def __init__(self, data_df = None, test = False, imsize = 256, load_images = False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            test: load testing images
            imsize: output image size
            load_images: load all images in memory. 256x256 images require 30GB memory.
                Please buy me some memory if you can.
        """
        super(ProteinDataset, self).__init__()
        self.test = test
        self.imsize = imsize
        self.images_path = test_images_path if test else train_images_path
        self.load_images = load_images
        if data_df is None:
            self.images_df = pd.read_csv(train_labels_path)
        else:
            self.images_df = data_df

        # Always preload targets in memory
        self.loaded_targets = torch.zeros(len(self.images_df), 28)
        for idx in range(len(self.images_df)):
            if self.test:
                targets = 0
            else:
                labels = self.images_df.loc[idx, 'Target']
                labels = [int(label) for label in labels.split()]
                self.loaded_targets[idx,labels] = 1

        # Preloading images in memory
        if self.load_images:
            self.loaded_images = torch.Tensor(len(self.images_df), 4, 
                self.imsize, self.imsize)
            for idx in range(len(self.images_df)):
                self.loaded_images[idx,:,:,:], _ = self.__getitem__(idx)

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        if self.load_images == True:
            image = self.loaded_images[idx,:,:,:]
        else:
            imagename = self.images_df.loc[idx, 'Id']
            image = np.zeros((4, self.imsize, self.imsize))
            for ch, channel in enumerate(color_channels):
                imagepath = self.images_path + imagename + '_' + channel + ".png"
                img = io.imread(imagepath)
                img = transform.resize(img, (self.imsize, self.imsize))
                image[ch,:,:] = img
            image /= 255.
            image = torch.from_numpy(image)
        targets = self.loaded_targets[idx,:]    
        return image.float(), targets
        
def get_data_loaders(imsize=256, batch_size=16):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(train_labels_path)
    train_df, valid_df = train_test_split(images_df, test_size=0.15, random_state=42)
    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()

    # set up the datasets
    train_dataset = ProteinDataset(data_df = train_df, imsize=imsize)
    valid_dataset = ProteinDataset(data_df = valid_df, imsize=imsize)

    train_sampler = SubsetRandomSampler(train_df.index)
    valid_sampler = SubsetRandomSampler(valid_df.index) 

    # set up the data loaders
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   sampler=train_sampler,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=False)

    valid_loader = DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   sampler=valid_sampler,
                                   num_workers=4,
                                   pin_memory=True)

    return train_loader, valid_loader

def get_test_loader(imsize=256, batch_size=128):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(test_submission_path)

    # set up the datasets
    test_dataset = ProteinDataset(data_df = images_df, imsize=imsize, test=True)

    test_sampler = SubsetRandomSampler(images_df.index)

    # set up the data loaders
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   sampler=test_sampler,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=False)

    return test_loader