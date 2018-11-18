import os, sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models.densenet import Atlas_DenseNet
from models.resnet import ResNet

from utils.dataloader import get_data_loaders, get_test_loader
from utils.misc import save_pred
from utils.metrics import accuracy, macro_f1

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
            valid_imgs = data[0].float().to(device)
            if not test:
                valid_labels = data[1].float().to(device)

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

def find_threshold(net, config, plot=False):
    print('Finding best threshold...')

    net.eval()

    test_loader, _ = get_data_loaders(imsize=config.imsize,
                                    batch_size=config.batch_size, test_size=0.)

    val_preds, val_labels = generate_preds(net, test_loader)

    f1s = []
    ths = []
    for th in np.arange(-0.5, 0.5, 0.05):
        th_vf1 = macro_f1(val_preds.numpy()>th, val_labels.numpy())
        f1s.append(th_vf1)
        ths.append(th)
    f1s = np.array(f1s)
    ths = np.array(ths)
    best_th = ths[f1s.argmax()]

    if plot == True:
        plt.plot(ths, f1s, 'bx')
        plt.plot(best_th, f1s.max(), 'ro')
        plt.ylabel('Macro F1s over testing set')
        plt.xlabel('Thresholds')
        plt.title('Threshold vs Macro F1')
        plt.show()

    print('Best Threshold: {:.2}, Best Eval Macro F1: {:.4}'.
        format(best_th, f1s.max()))
    return best_th

def generate_submission(net, config, folds=1, SUBM_OUT=None):
    print('Generating submission...')

    if SUBM_OUT is None:
        model_params = [config.model_name, config.exp_name]
        SUBM_OUT = './subm/best_{}_{}.csv'.format(*model_params)
    print('Saving to ', SUBM_OUT)

    net.eval()
    
    best_th = find_threshold(net, config)

    test_loader = get_test_loader(imsize=config.imsize, 
                                    batch_size=config.batch_size)

    test_preds = torch.zeros(len(test_loader.dataset), 28)
    for _ in range(folds):
        test_preds += generate_preds(net, test_loader, test=True)
    test_preds = test_preds.numpy()/float(folds)

    preds_df = pd.DataFrame(data=test_preds)
    preds_df['th'] = best_th
    preds_df.to_csv(SUBM_OUT.replace('subm', 'preds'), index=False)

    save_pred(test_preds, best_th, SUBM_OUT)