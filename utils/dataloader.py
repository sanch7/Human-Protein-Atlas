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
external_labels_path = f"./data/external/HPAv18RBGY_wodpl.csv"
test_submission_path = f"./data/sample_submission.csv"
train_images_path = f"./data/train/"
external_images_path = f"./data/external/images512/"
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

def load_image(id, dataset = "train", colors = color_channels):
    npy_path = './data/{}_npy/'.format(dataset)
    if dataset == "external":
        dataset = "train"
    # if os.path.exists(npy_path + '{}.npy'.format(id)):
    try:
        image = np.load(npy_path + '{}.npy'.format(id))
        if len(colors) == 3:
            image = image[:,:,:3]

    # # else:
    except:
        # print('Loading raw images')
        image = np.zeros((512, 512, len(colors)), dtype='uint8')
        for ch, channel in enumerate(colors):
            # if dataset == "external":
            #     imagepath = './data/{}/images512/'.format(dataset) + id + '_' + channel + ".jpg"
            # else:
            imagepath = './data/{}/'.format(dataset) + id + '_' + channel + ".png"
            img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                imagepath = './data/{}/'.format(dataset) + id + '_' + channel + ".jpg"
                img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            image[:,:, ch] = img
    return image

def mixup_loader(idx, df, dataset, colors):
    mixid = df.sample()
    # if dataset=="train":
    #     print(mixid)
    ratio = np.random.rand()

    targets1 = df.loc[idx, 'Target']
    targets2 = mixid['Target'].values[0]
    # print("Target1, ", targets1, type(targets1), dataset)
    # print("Target2, ", targets2, type(targets2), dataset)
    targets = ratio*targets1 + (1-ratio)*targets2

    image1 = load_image(df.loc[idx, 'Id'], dataset, colors)
    image2 = load_image(mixid['Id'].values[0], dataset, colors)
    image = (ratio*image1 + (1-ratio)*image2).round().astype('uint8')
    # print("ids = {}, {}. Ratio = {}".format(df.loc[idx, 'Id'], mixid[0], ratio))
    return image, targets

class ProteinDataset(Dataset):
    def __init__(self, data_df=None, test=False, imsize=256, num_channels=4, 
                    transformer=None, preload=False, mixup=False):
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
        self.dataset = "test" if test else "train"
        self.mixup = mixup
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
        if self.mixup:
            image, targets = mixup_loader(idx, self.images_df, self.dataset, self.colors)

        else:
            if self.preload:
                image = self.imarray[idx,:,:,:]
            else:
                image = load_image(imagename, self.dataset, self.colors)
            targets = self.images_df['Target'][idx]    
        
        if self.transformer:
            image = self.transformer(image=image)['image']

        else:
            image = transform.resize(image, (self.imsize, self.imsize))

        image = torch.from_numpy(image).permute(-1, 0, 1).float()
        return image, targets, imagename

    def cmix(self, idx):
        return mixup_loader(idx, self.images_df, self.dataset, self.colors)

    def getImageName(self, imagename):
        image = np.zeros((512, 512, len(self.colors)), dtype='uint8')
        for ch, channel in enumerate(self.colors):
            imagepath = self.images_path + imagename + '_' + channel + ".png"
            img = Image.open(imagepath)
            image[:,:, ch] = img

        targets = self.images_df[self.images_df['Id'] == imagename]['Target'].values[0]   
        return image, targets
        
def get_data_loaders(imsize=256, num_channels=4, batch_size=16, test_size=0.15, num_workers=4, 
                        preload=False, eval_mode=False, external_data=False, mixup=False):
    '''sets up the torch data loaders for training'''
    images_df = pd.read_csv(train_labels_path)
    # train_df, valid_df = train_test_split(images_df, test_size=test_size, random_state=42)
    images_df['labels'] = images_df['Target'].apply(label_gen_np)
    train_idx, _, valid_idx, _ = iterative_train_test_split(np.arange(len(images_df))[:, None], 
                                    np.stack(images_df['labels']), test_size=test_size,
                                    random_state=21)
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
    train_dataset = ProteinDataset(data_df=train_df, imsize=imsize, 
                                    num_channels=num_channels, 
                                    transformer=train_tf, preload=preload,
                                    mixup=mixup)
    valid_dataset = ProteinDataset(data_df = valid_df, imsize=imsize, 
                                    num_channels = num_channels, 
                                    transformer = valid_tf, preload=preload)

    if external_data and not eval_mode:
        external_dataset = ProteinExternalDataset(imsize=imsize, 
                                    num_channels = num_channels,
                                    transformer = train_tf, preload=preload,
                                    mixup=mixup)
        
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

def get_preds_loader(imsize=256, num_channels=4, batch_size=16, num_workers=4, 
                        preload=False, mixup=False):
    '''sets up the torch data loaders for training'''
    train_df = pd.read_csv(train_labels_path)
    train_df['labels'] = train_df['Target'].apply(label_gen_np)

    # set up the transformers
    train_tf = alb_transform_test(imsize, num_channels)
    # train_tf = train_transformer(imsize)
    # valid_tf = test_transformer(imsize)

    # set up the datasets
    train_dataset = ProteinDataset(data_df=train_df, imsize=imsize, 
                                    num_channels=num_channels, 
                                    transformer=train_tf, preload=preload,
                                    mixup=mixup)

    external_dataset = ProteinExternalDataset(imsize=imsize, 
                                    num_channels = num_channels,
                                    transformer = train_tf, preload=preload,
                                    mixup=mixup)
    # set up the data loaders
    train_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    external_loader = DataLoader(external_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    test_df = pd.read_csv(test_submission_path)

    # set up the transformer
    test_tf = alb_transform_test(imsize, num_channels)

    # set up the datasets
    test_dataset = ProteinDataset(data_df = test_df, imsize=imsize, 
                                    num_channels = num_channels,
                                    transformer = test_tf, test=True)

    # set up the data loaders
    test_loader = DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   drop_last=False)

    return train_loader, external_loader, test_loader

class ProteinExternalDataset(Dataset):
    def __init__(self, data_df=None, imsize=256, num_channels=4, 
                    transformer=None, preload=False, mixup=False):
        """
        Params:
            data_df: data DataFrame of image name and labels
            imsize: output image size
        """
        super(ProteinExternalDataset, self).__init__()
        self.imsize = imsize
        self.transformer = transformer
        self.preload = preload
        self.colors = color_channels[:num_channels]
        self.images_path = external_images_path
        self.mixup = mixup
        self.dataset = "external"
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
                                        self.imsize, len(self.colors)), dtype='uint8')
            for idx, imagename in enumerate(tqdm(self.images_df['Id'])):
                for ch, channel in enumerate(self.colors):
                    imagepath = self.images_path + imagename + '_' + channel + ".jpg"
                    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (self.imsize, self.imsize), 
                                        interpolation=cv2.INTER_AREA)
                    self.imarray[idx,:,:,ch] = img

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Id']
        if self.mixup:
            image, targets = mixup_loader(idx, self.images_df, self.dataset, self.colors)

        else:
            if self.preload:
                image = self.imarray[idx,:,:,:]
            else:
                image = load_image(imagename, self.dataset, self.colors)
            targets = self.images_df['Target'][idx]    

        if self.transformer:
            image = self.transformer(image=image)['image']
        else:
            image = transform.resize(image, (self.imsize, self.imsize))

        image = torch.from_numpy(image).permute(-1, 0, 1).float()
        return image, targets, str(imagename)