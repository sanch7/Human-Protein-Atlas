import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def save_pred(pred, th=0., SUBM_OUT='./subm/submission.csv'):
    pred_list = []
    for line in pred:
        s = ' '.join(list([str(i) for i in np.nonzero(line>=th)[0]]))
        if s == '':
            s = str(line.argmax())
        pred_list.append(s)
    
    sample_df = pd.read_csv('./data/sample_submission.csv')
    sample_df['Predicted'] = pred_list
    sample_df.to_csv(SUBM_OUT, header=True, index=False)
    print('Saved to ', SUBM_OUT)

def log_metrics(train_losses, valid_losses, valid_f1s, lr_hist, e, model_ckpt):
    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].plot(train_losses)
    axes[0, 1].plot(valid_losses)
    axes[1, 0].plot(valid_f1s)
    axes[1, 1].plot(lr_hist)
    axes[0, 0].set_title('Train Loss')
    axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_title('Val F1')
    axes[1, 1].set_title('LR History')
    plt.suptitle("At Epoch {}".format(e+1), fontsize=16)
    plt.savefig(model_ckpt.replace('model_weights', 'logs').replace('.pth', '.png'))