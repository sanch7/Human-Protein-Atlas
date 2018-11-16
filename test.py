import os
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

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--imsize', default=256, type=int, 
                    help='image size')
parser.add_argument('--batch_size', default=200, type=int, 
                    help='size of batches')
parser.add_argument('--model_name', default='resnet', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='run5_FocalLoss', type=str,
                    help='name of experiment for saving files')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = './model_weights/best_{}_{}.pth'.format(args.model_name,
                                                       args.exp_name)
OUT_FILE = './subm/' + os.path.basename(model_path.replace('pth', 'csv'))
test_submission_path = f"./data/sample_submission.csv"
print('Saving to {}'.format(OUT_FILE))

def test(net, optimizer, loss, test_loader):
    net.eval() 
    #keep track of preds
    val_labels = torch.Tensor(len(valid_loader.dataset), 28)
    val_preds = torch.Tensor(len(valid_loader.dataset), 28)
    ci = 0

    t0 = time.time()
    ll = len(train_loader)
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            valid_imgs = data[0].to(device)
            valid_labels = data[1].to(device)
            
            # get predictions
            label_vpreds = net(valid_imgs)
            val_preds[ci: ci+label_vpreds.shape[0], :] = label_vpreds
            val_labels[ci: ci+valid_labels.shape[0], :] = valid_labels
            ci = ci+label_vpreds.shape[0]

            # make a cool terminal output
            tc = time.time() - t0
            tr = int(tc*(ll-i-1)/(i+1))
            sys.stdout.write('\r')
            sys.stdout.write('B: {:>3}/{:<3} | Loss: {:.4} | ETA: {:d}s'.
                format(i+1, ll, tloss.item(), tr))

    val_preds = val_preds > 0
    return val_preds.numpy(), val_labels.numpy()

if __name__ == '__main__':
    test()
    

    