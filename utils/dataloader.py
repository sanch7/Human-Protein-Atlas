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
from skmultilearn.model_selection import iterative_train_test_split

from skimage import io, transform
import cv2
from .preprocessing import alb_transform_train, alb_transform_test, custom_over_sampler
from .misc import label_gen_tensor, label_gen_np

import warnings
warnings.filterwarnings("ignore")

train_labels_path = f"./data/train.csv"
external_labels_path = f"./data/subcellular/augment.csv"
test_submission_path = f"./data/sample_submission.csv"
train_images_path = f"./data/train/"
external_images_path = f"./data/subcellular/images/"
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
    def __init__(self, data_df = None, test = False, imsize = 256, 
                    num_channels = 4, transformer = None, preload=False):
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
        self.colors = color_channels[:num_channels]
        self.images_path = test_images_path if test else train_images_path
        if data_df is None:
            self.images_df = pd.read_csv(train_labels_path)
        else:
            if 'Target' not in data_df.columns:
                data_df['Target'] = torch.zeros(len(data_df), 28)
            self.images_df = data_df

        if not self.test:
            self.images_df['Target'] = self.images_df['Target'].apply(label_gen_tensor)

        if preload:
            print('Preloading images...')
            self.imarray = np.zeros((len(self.images_df), self.imsize, 
                                        self.imsize, len(self.colors)), dtype='uint8')
            for idx, imagename in enumerate(tqdm(self.images_df['Id'])):
                for ch, channel in enumerate(self.colors):
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
            image = np.zeros((512, 512, len(self.colors)), dtype='uint8')
            for ch, channel in enumerate(self.colors):
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
        image = np.zeros((512, 512, len(self.colors)), dtype='uint8')
        for ch, channel in enumerate(self.colors):
            imagepath = self.images_path + imagename + '_' + channel + ".png"
            img = Image.open(imagepath)
            image[:,:, ch] = img

        targets = self.images_df[self.images_df['Id'] == imagename]['Target']    
        return image, targets
        
def get_data_loaders(imsize=256, num_channels=4, batch_size=16, test_size=0.15, num_workers=4, 
                        preload=False, eval_mode=False, external_data=False):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(train_labels_path)
    # train_df, valid_df = train_test_split(images_df, test_size=test_size, random_state=42)
    images_df['labels'] = images_df['Target'].apply(label_gen_np)
    valid_idx, _, train_idx, _ = iterative_train_test_split(np.arange(len(images_df))[:, None], 
                                    np.stack(images_df['labels']), test_size=test_size)
    train_df = images_df.loc[train_idx.squeeze(1)]
    valid_df = images_df.loc[valid_idx.squeeze(1)]

    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()

    # Oversampling
    # if not test_size == 0:
    #     train_df = custom_over_sampler(train_df, factor=2, num_classes=10)

    # set up the transformers
    if eval_mode:
        train_tf = alb_transform_test(imsize, num_channels)
    else:
        train_tf = alb_transform_train(imsize, num_channels)
    valid_tf = alb_transform_test(imsize, num_channels)
    # train_tf = train_transformer(imsize)
    # valid_tf = test_transformer(imsize)

    # set up the datasets
    train_dataset = ProteinDataset(data_df = train_df, imsize=imsize, 
                                    num_channels = num_channels, 
                                    transformer = train_tf, preload=preload)
    valid_dataset = ProteinDataset(data_df = valid_df, imsize=imsize, 
                                    num_channels = num_channels, 
                                    transformer = valid_tf, preload=preload)

    external_dataset = ProteinExternalDataset(imsize=imsize, 
                                    transformer = train_tf, preload=preload)

    if external_data and not eval_mode:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, external_dataset])
        
    train_sampler = SubsetRandomSampler(range(len(train_dataset)))
    valid_sampler = SubsetRandomSampler(range(len(valid_dataset))) 

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

def get_test_loader(imsize=256, num_channels=4, batch_size=16, num_workers=4):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(test_submission_path)

    # set up the transformer
    test_tf = alb_transform_test(imsize, num_channels)

    # set up the datasets
    test_dataset = ProteinDataset(data_df = images_df, imsize=imsize, 
                                    num_channels = num_channels,
                                    transformer = test_tf, test=True)

    # set up the data loaders
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return test_loader

class ProteinExternalDataset(Dataset):
    def __init__(self, data_df = None, imsize = 256, 
                    transformer = None, preload=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super(ProteinExternalDataset, self).__init__()
        self.imsize = imsize
        self.transformer = transformer
        self.preload = preload
        self.images_path = external_images_path
        if data_df is None:
            self.images_df = pd.read_csv(external_labels_path)
            self.images_df.columns=['Id', 'Target']
        else:
            if 'Target' not in data_df.columns:
                data_df['Target'] = torch.zeros(len(data_df), 28)
            self.images_df = data_df

        self.images_df['Target'] = self.images_df['Target'].apply(label_gen_tensor)

        if preload:
            print('Preloading images...')
            self.imarray = np.zeros((len(self.images_df), self.imsize, 
                                        self.imsize, 3), dtype='uint8')
            for idx, imagename in enumerate(tqdm(self.images_df['Id'])):
                imagepath = self.images_path + str(imagename) + "_rgb.jpg"
                img = cv2.imread(imagepath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.imsize, self.imsize), 
                                    interpolation=cv2.INTER_AREA)
                self.imarray[idx,:,:,:] = img

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Id']
        if self.preload:
            image = self.imarray[idx,:,:,:]
        else:
            imagepath = self.images_path + str(imagename) + "_rgb.jpg"
            image = cv2.imread(imagepath, cv2.IMREAD_COLOR)

        if self.transformer:
            image = self.transformer(image=image)['image']
        else:
            image = transform.resize(image, (self.imsize, self.imsize))
        image = torch.from_numpy(image).permute(-1, 0, 1).float()
        targets = self.images_df['Target'][idx]    
        return image, targets, str(imagename)

class ProteinMergedDataset(Dataset):
    def __init__(self, data_df = None, test = False, imsize = 256, 
                    num_channels = 4, transformer = None, preload=False, 
                    external_data=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super(ProteinMergedDataset, self).__init__()
        self.test = test
        self.imsize = imsize
        self.transformer = transformer
        self.preload = preload
        self.external_data = external_data
        num_channels = 3 if self.external_data else num_channels
        self.colors = color_channels[:num_channels]
        self.images_path = test_images_path if test else train_images_path
        self.ex_images_path = external_images_path
        if data_df is None:
            self.images_df = pd.read_csv(train_labels_path)
        else:
            if 'Target' not in data_df.columns:
                data_df['Target'] = torch.zeros(len(data_df), 28)
            self.images_df = data_df

        if self.external_data:
            self.ex_images_df = pd.read_csv(external_labels_path)
            self.ex_images_df.columns=['Id', 'Target']
            self.ex_images_df['Target'] = self.ex_images_df['Target'].apply(label_gen_tensor)
            
        if not self.test:
            self.images_df['Target'] = self.images_df['Target'].apply(label_gen_tensor)

        if preload:
            print('Preloading images...')
            self.imarray = np.zeros((len(self.images_df), self.imsize, 
                                        self.imsize, len(self.colors)), dtype='uint8')
            for idx, imagename in enumerate(tqdm(self.images_df['Id'])):
                for ch, channel in enumerate(self.colors):
                    imagepath = self.images_path + imagename + '_' + channel + ".png"
                    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.imsize, self.imsize), 
                                        interpolation=cv2.INTER_AREA)
                    self.imarray[idx,:,:,ch] = img

    def __len__(self):
        if not self.external_data:
            return len(self.images_df)
        else:
            return len(self.images_df) + len(self.ex_images_df)

    def __getitem__(self, idx):
        if idx < len(self.images_df):
            imagename = self.images_df.loc[idx, 'Id']
            targets = self.images_df['Target'][idx]    
            ex_flag = False
        else:
            idx -= len(self.images_df)
            imagename = self.ex_images_df.loc[idx, 'Id']
            targets = self.ex_images_df['Target'][idx]    
            ex_flag = True

        if not ex_flag and self.preload:
            image = self.imarray[idx,:,:,:]
        elif not ex_flag:
            image = np.zeros((512, 512, len(self.colors)), dtype='uint8')
            for ch, channel in enumerate(self.colors):
                imagepath = self.images_path + imagename + '_' + channel + ".png"
                img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)     #232s
                # img = io.imread(imagepath)                            #239s
                # img = Image.open(imagepath)                           #236s
                image[:,:, ch] = img
        else:
            imagename = str(imagename)
            imagepath = self.ex_images_path + imagename + "_rgb.jpg"
            image = cv2.imread(imagepath, cv2.IMREAD_COLOR)

        if self.transformer:
            image = self.transformer(image=image)['image']
        else:
            image = transform.resize(image, (self.imsize, self.imsize))
        image = torch.from_numpy(image).permute(-1, 0, 1).float()
        return image, targets, imagename