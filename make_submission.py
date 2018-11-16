import os, sys
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

from utils.dataloader import get_data_loaders, get_test_loader
from utils.misc import str2bool
from evaluations import validate, generate_submission
from utils.metrics import FocalLoss, accuracy, macro_f1

import models.model_list as model_list

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', default='./configs/config.json', 
                    help="Run configuration")
parser.add_argument('-g', '--generate', type=str2bool, nargs='?',
                    const=True, default='1', 
                    help="Generate submission or validate. [0/1]")
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

if not os.path.exists('./subm'):
    os.makedirs('./subm')

def main_eval():
    model_params = [config.model_name, config.exp_name]
    MODEL_CKPT = './model_weights/best_{}_{}.pth'.format(*model_params)
    print('Loading model from ' + MODEL_CKPT)

    Net = getattr(model_list, config.model_name)
    
    net = Net()
    net = nn.parallel.DataParallel(net)
    net.to(device)

    net.load_state_dict(torch.load(MODEL_CKPT))

    if args.generate:
        generate_submission(net, config)
    else:
        validate(net, config)

if __name__ == '__main__':
    main_eval()
