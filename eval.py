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
from generate_preds import generate_preds
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

def save_pred(pred, th=0.5, fname='./subm/submission.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
    
    sample_df = pd.read_csv('./data/sample_submission.csv')
    sample_df['Predicted'] = pred_list
    sample_df.to_csv(fname, header=True, index=False)
    print('Saved to ', fname)

def validate(net):
    print('Validating model...')
    net.eval()
    _, valid_loader = get_data_loaders(imsize=config.imsize,
                                            batch_size=config.batch_size)

    val_preds, val_labels = generate_preds(net, valid_loader)
    
    epoch_vf1 = macro_f1(val_preds.numpy()>0, val_labels.numpy())
    epoch_vacc = accuracy(val_preds.numpy()>0, val_labels.numpy())
    print('Avg Eval Macro F1: {:.4}, Avg Eval Acc. {:.4}'.
        format(epoch_vf1, epoch_vacc))

def generate_submission(net):
    print('Generating submission...')
    net.eval()
    
    test_loader = get_test_loader(imsize=config.imsize, 
                                    batch_size=config.batch_size)

    test_preds = generate_preds(net, test_loader, test=True)

    model_params = [config.model_name, config.exp_name]
    MODEL_OUT = './subm/best_{}_{}.pth'.format(*model_params)

    save_pred(test_preds, 0., MODEL_OUT)

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
        generate_submission(net)
    else:
        validate(net)

if __name__ == '__main__':
    main_eval()
