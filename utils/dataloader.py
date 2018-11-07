from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms

import warnings
warnings.filterwarnings("ignore")

train_labels_path = f"./data/train.csv"
train_images_path = f"./data/train/"
test_images_path = f"./data/test/"

labels_dict={ 0: "Nucleoplasm", 1: "Nuclear membrane", 2: "Nucleoli", 3: "Nucleoli fibrillar center", 4: "Nuclear speckles", 5: "Nuclear bodies", 6: "Endoplasmic reticulum", 7: "Golgi apparatus", 8: "Peroxisomes", 9: "Endosomes", 10: "Lysosomes", 11: "Intermediate filaments", 12: "Actin filaments", 13: "Focal adhesion sites", 14: "Microtubules", 15: "Microtubule ends", 16: "Cytokinetic bridge", 17: "Mitotic spindle", 18: "Microtubule organizing center", 19: "Centrosome", 20: "Lipid droplets", 21: "Plasma membrane", 22: "Cell junctions", 23: "Mitochondria", 24: "Aggresome", 25: "Cytosol", 26: "Cytoplasmic bodies", 27: "Rods & rings" }

color_channels = ('red','green','blue','yellow')

class ProteinDataset(Dataset):
	def __init__(self, test = False):
		super(ProteinDataset, self).__init__()
		self.test = test
		self.images_path = test_images_path if test else train_images_path
		self.images_df = pd.read_csv(train_labels_path)

	def __len__(self):
		return len(self.images_df)

	def __getitem__(self, idx):
		imagename = self.images_df.loc[idx, 'Id']
		image = []
		for channel in color_channels:
			imagepath = self.images_path + imagename + '_' + channel + ".png"
			image.append(io.imread(imagepath))
		image = torch.Tensor(image).permute(1, 2, 0)
		image = image/255
		labels = self.images_df.loc[idx, 'Target']
		labels = sorted(labels.split())
		sample = {"image": image, "labels": labels}
		return sample