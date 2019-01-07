import os, sys
import glob
import argparse
import time
import json
import pprint
from collections import namedtuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils.dataloader import get_preds_loader
from evaluations import generate_submission, generate_preds
from utils.metrics import FocalLoss, accuracy, macro_f1
from utils.misc import save_pred
from apply_leak import apply_leak

import models.model_list as model_list

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', default='./configs/config.json', 
                    help="Run configuration")
parser.add_argument('--outfile', default='', 
                    help="Append arg to the file name")
parser.add_argument('-f', '--folds', type=int, default=1, 
                    help="Number of folds for predictions averaging")
parser.add_argument('--submission', action='store_true', default=False,
                    help='Generate submission')
args = parser.parse_args()

with open(args.config) as f_in:
    d = json.load(f_in)
    config = namedtuple("config", d.keys())(*d.values())

print("Loaded configuration from ", args.config)
print('')
pprint.pprint(d)
time.sleep(5)
print('')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True

if not os.path.exists('./preds'):
    os.makedirs('./preds')

def main_pred(net = None, opcon = None, attn=False):
    if opcon is not None:
        config = opcon

    model_params = [config.model_name, config.exp_name]
    MODEL_CKPT = './model_weights/best_{}_{}.pth'.format(*model_params)

    if net is None:
        Net = getattr(model_list, config.model_name)
        net = Net(num_channels = config.num_channels)
        net = nn.parallel.DataParallel(net)
        net.to(device)

    print('Loading model from ' + MODEL_CKPT)

    try:
        net.load_state_dict(torch.load(MODEL_CKPT))
    except:
        net.load_state_dict(torch.load(MODEL_CKPT)['state_dict'])


    PRED_OUT = './preds/best_{}_{}_train.csv'.format(*model_params)
    if args.outfile != '':
        PRED_OUT = PRED_OUT.replace('.csv', '_{}.csv'.format(args.outfile))

    print('Generating predictions...')

    net.eval()

    tload, eload, testload = get_preds_loader(imsize=config.imsize, 
                                    num_channels=config.num_channels,  
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers)

    # test_preds = torch.zeros(len(test_loader.dataset), 28)
    # for _ in range(folds):
    #     test_preds += generate_preds(net, test_loader, test=True, attn=attn)
    # test_preds = test_preds.numpy()/float(folds)
    testpreds = generate_preds(net, testload, test=True, attn=attn)
    pd.DataFrame(data=testpreds.numpy()).to_csv(PRED_OUT.replace('train', 'test'),
                                                index=False)

    tpreds = generate_preds(net, tload, test=True, attn=attn)
    pd.DataFrame(data=tpreds.numpy()).to_csv(PRED_OUT, index=False)

    epreds = generate_preds(net, eload, test=True, attn=attn)
    pd.DataFrame(data=epreds.numpy()).to_csv(PRED_OUT.replace('train', 'external'),
                                                index=False)


if __name__ == '__main__':
    main_pred(opcon=config)