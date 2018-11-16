import os, sys
import argparse
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.densenet import Atlas_DenseNet
from models.resnet import ResNet

from utils.dataloader import get_test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_preds(net, test_loader, test=False):
    net.eval() 
    
    if not test:
        val_labels = torch.Tensor(len(test_loader.dataset), 28)
    val_preds = torch.Tensor(len(test_loader.dataset), 28)
    ci = 0

    t0 = time.time()
    ll = len(test_loader)
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            valid_imgs = data[0].to(device)
            if not test:
                valid_labels = data[1].to(device)
            
            # get predictions
            label_vpreds = net(valid_imgs)
            val_preds[ci: ci+label_vpreds.shape[0], :] = label_vpreds
            if not test:
                val_labels[ci: ci+valid_labels.shape[0], :] = valid_labels
            ci = ci+label_vpreds.shape[0]

            # make a cool terminal output
            tc = time.time() - t0
            tr = int(tc*(ll-i-1)/(i+1))
            sys.stdout.write('\r')
            sys.stdout.write('B: {:>3}/{:<3} | ETA: {:>4d}s'.
                format(i+1, ll, tr))
    print('')

    if not test:
        return val_preds, val_labels
    else:
        return val_preds