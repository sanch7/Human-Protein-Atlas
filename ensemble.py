import os, sys
import glob
import argparse
import time
from datetime import datetime
import json
import pprint
from collections import namedtuple

import pandas as pd
import numpy as np

from utils.misc import save_pred, label_gen_np

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', default='./configs/config.json', 
                    help="Run configuration")
parser.add_argument('--outfile', default='', 
                    help="Append arg to the file name")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-p', '--preds', action='store_true', default=False,
                    help="Ensemble predictions in the preds folder")
group.add_argument('-s', '--subm', action='store_true', default=False,
                    help="Ensemble submissions in the subm folder")
args = parser.parse_args()

with open(args.config) as f_in:
    d = json.load(f_in)
    config = namedtuple("config", d.keys())(*d.values())

print("Loaded configuration from ", args.config)
print('')
pprint.pprint(d)
time.sleep(5)
print('')

if not os.path.exists('./subm'):
    os.makedirs('./subm')

def preds_ensemble():
    if len(os.listdir('./preds/') ) == 0:
        raise ValueError('Preds directory is empty')

    all_preds = np.zeros((len(pd.read_csv('./data/sample_submission.csv')), 28))
    th = 0

    for i, filepath in enumerate(glob.iglob('./preds/*.csv')):
        print('Processing file', filepath.split('/')[-1])
        predi = pd.read_csv(filepath)
        all_preds += predi[predi.columns[:-1]]
        th += predi['th'][0]
    all_preds /= float(i+1)
    th /= float(i+1)

    SUBM_OUT = './subm/pred_ensemble_{}.csv'.\
                        format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    if args.outfile != '':
        SUBM_OUT = SUBM_OUT.replace('.csv', '_{}.csv'.format(args.outfile))

    save_pred(all_preds, th, SUBM_OUT)

def subm_ensemble():
    if len(os.listdir('./subm/') ) == 0:
        raise ValueError('Submission directory is empty')

    all_preds = np.zeros((len(pd.read_csv('./data/sample_submission.csv')), 28))
    
    for i, filepath in enumerate(glob.iglob('./subm/*.csv')):
        print('Processing file', filepath.split('/')[-1])
        predi = pd.read_csv(filepath)
        all_preds += np.stack(predi['Predicted'].apply(label_gen_np)).astype(np.float)
    
    SUBM_OUT = './subm/subm_ensemble_{}.csv'.\
                        format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    if args.outfile != '':
        SUBM_OUT = SUBM_OUT.replace('.csv', '_{}.csv'.format(args.outfile))

    save_pred(all_preds, float(i+1)/2., SUBM_OUT)

if __name__ == '__main__':
    if args.preds:
        preds_ensemble()
    elif args.subm:
        subm_ensemble()
    else:
        print("No input given. Please provide [p/s] in args.")