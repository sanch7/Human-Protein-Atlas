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
from models.resnet import Atlas_ResNet

from utils.dataloader import get_data_loaders, get_test_loader
from utils.misc import save_pred
from utils.metrics import accuracy, macro_f1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labels_dict={ 0: "Nucleoplasm", 1: "Nuclear membrane", 2: "Nucleoli", 
    3: "Nucleoli fibrillar center", 4: "Nuclear speckles", 5: "Nuclear bodies", 
    6: "Endoplasmic reticulum", 7: "Golgi apparatus", 8: "Peroxisomes", 
    9: "Endosomes", 10: "Lysosomes", 11: "Intermediate filaments", 
    12: "Actin filaments", 13: "Focal adhesion sites", 14: "Microtubules", 
    15: "Microtubule ends", 16: "Cytokinetic bridge", 17: "Mitotic spindle", 
    18: "Microtubule organizing center", 19: "Centrosome", 20: "Lipid droplets", 
    21: "Plasma membrane", 22: "Cell junctions", 23: "Mitochondria", 
    24: "Aggresome", 25: "Cytosol", 26: "Cytoplasmic bodies", 27: "Rods & rings"}

def generate_preds(net, test_loader, test=False, attn=False):
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
            if not attn:
                label_vpreds = net(valid_imgs)
            else:
                label_vpreds, _ = net(valid_imgs)

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

def generate_submission(net, config, folds=1, SUBM_OUT=None, gen_csv=True, attn=False):
    print('Generating predictions...')

    net.eval()

    test_loader = get_test_loader(imsize=config.imsize, 
                                    num_channels=config.num_channels,  
                                    batch_size=config.batch_size)

    test_preds = torch.zeros(len(test_loader.dataset), 28)
    for _ in range(folds):
        test_preds += generate_preds(net, test_loader, test=True, attn=attn)
    test_preds = test_preds.numpy()/float(folds)

    if gen_csv:
        print('Generating submission with class wise thresholding...')
        best_th = find_threshold(net, config, class_wise=True, plot=True)

        preds_df = pd.DataFrame(data=test_preds)
        preds_df['th'] = pd.Series(best_th)
        preds_df.to_csv(SUBM_OUT.replace('subm', 'preds'), index=False)

        save_pred(test_preds, best_th, SUBM_OUT)

    return test_preds

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "F1={:.3f}\nTh={:.3f}".format(ymax, xmax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(xycoords='data',textcoords="axes fraction",
                bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

def find_threshold(net, config, class_wise=True, plot=True, attn=False):
    print('Finding best threshold...')

    net.eval()

    test_loader, _ = get_data_loaders(imsize=config.imsize,
                                    num_channels=config.num_channels,  
                                    batch_size=config.batch_size, test_size=0.,
                                    num_workers=config.num_workers, eval_mode=True)

    val_preds, val_labels = generate_preds(net, test_loader, attn=attn)

    if class_wise:
        search_range = np.arange(-2., 1., 0.05)
        f1s = np.zeros((28, len(search_range)))
        ths = np.zeros((28, len(search_range)))
        for th_id, th in enumerate(search_range):
            th_vf1 = macro_f1(val_preds.numpy()>th, 
                                val_labels.numpy(), class_wise=True)
            f1s[:, th_id] = th_vf1
            ths[:, th_id] = th
        best_th = ths[range(ths.shape[0]), f1s.argmax(axis=1)]

        if plot == True:
            fig, axes = plt.subplots(7, 4, sharex='col', sharey='row', figsize=(16, 16))
            fig.suptitle("Threshold vs Class F1 score", fontsize=16)
            for i in range(28):
                axes[i // 4, i % 4].plot(ths[i ,:], f1s[i, :])
                axes[i // 4, i % 4].plot(best_th[i], f1s.max(axis=1)[i], 'ro')
                annot_max(ths[i ,:], f1s[i, :], axes[i // 4, i % 4])
                axes[i // 4, i % 4].set_title(labels_dict[i])
            fig.savefig('./logs/best_{}_{}_F1_vs_th.png'.format(config.model_name, 
                                config.exp_name))

    else:
        f1s = []
        ths = []
        for th in np.arange(-2., 1., 0.05):
            th_vf1 = macro_f1(val_preds.numpy()>th, val_labels.numpy())
            f1s.append(th_vf1)
            ths.append(th)
        f1s = np.array(f1s)
        ths = np.array(ths)
        best_th = ths[f1s.argmax()]

        if plot == True:
            plt.plot(ths, f1s, 'bx')
            plt.plot(best_th, f1s.max(), 'ro')
            annot_max(best_th, f1s.max(), plt)
            plt.ylabel('Macro F1s over testing set')
            plt.xlabel('Thresholds')
            plt.title('Threshold vs Macro F1')
            plt.savefig('./logs/best_{}_{}_F1_vs_th.png'.format(config.model_name, 
                                config.exp_name))

    if class_wise:
        print('Best Thresholds: ', best_th)
        print('Best Eval Macro F1: ', f1s.max(axis=1))
        print('Best Eval Macro F1 Avg: ', f1s.max(axis=1).mean())
    else:
        print('Best Threshold: {:.2}, Best Eval Macro F1: {:.4}'.
            format(best_th, f1s.max()))
    return best_th