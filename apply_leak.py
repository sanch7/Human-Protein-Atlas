import argparse
import json
import pprint
from collections import namedtuple

import numpy as np 
import pandas as pd 

bleakfile = './leaks/brian_overlap.csv'
mleakfile = './leaks/moriyama_leak.csv'

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', default='./configs/config.json', 
                    help="Run configuration")
args = parser.parse_args()

with open(args.config) as f_in:
    d = json.load(f_in)
    config = namedtuple("config", d.keys())(*d.values())

def apply_leak(submfile = None):
    if submfile is None:
        model_params = [config.model_name, config.exp_name]
        submfile = './subm/best_{}_{}.csv'.format(*model_params)

    submdf = pd.read_csv(submfile)
    bleakdf = pd.read_csv(bleakfile)
    mleakdf = pd.read_csv(mleakfile)

    for i, t in zip(mleakdf['Test'], mleakdf['Target']):
        submdf.loc[submdf['Id']==i, 'Predicted'] = t

    for i, t in zip(bleakdf['Test'], bleakdf['Target']):
        submdf.loc[submdf['Id']==i, 'Predicted'] = t

    submfile = submfile.replace('.csv', '_m_b.csv')
    submdf.to_csv(submfile, header=True, index=False)
    print('Saved to ', submfile)

if __name__ == '__main__':
    apply_leak()