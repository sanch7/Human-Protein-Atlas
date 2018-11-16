import argparse

import numpy as np
import pandas as pd

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_pred(pred, th=0.5, fname='./subm/submission.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>th)[0]]))
        pred_list.append(s)
    
    sample_df = pd.read_csv('./data/sample_submission.csv')
    sample_df['Predicted'] = pred_list
    sample_df.to_csv(fname, header=True, index=False)
    print('Saved to ', fname)