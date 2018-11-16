import os, sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from models.densenet import Atlas_DenseNet
from models.uselessnet import UselessNet
from models.resnet import ResNet
from utils.dataloader import get_data_loaders
from utils.metrics import FocalLoss, accuracy, macro_f1

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--dev_mode', action='store_true', default=False,
                    help='train only few batches per epoch')
parser.add_argument('--imsize', default=256, type=int, 
                    help='image size')
parser.add_argument('--batch_size', default=16, type=int, 
                    help='size of batches')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of epochs')
parser.add_argument('--lr', default=0.001, type=float,
                    help='initial learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float,
                    help='l2 regularization for model')
parser.add_argument('--l2', default=1e-4, type=float,
                    help='l2 regularization for model')
parser.add_argument('--es_patience', default=50, type=int, 
                    help='early stopping patience')
parser.add_argument('--model_name', default='densenet121', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='runtest_FocalLoss', type=str,
                    help='name of experiment for saving files')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists('./model_weights'):
    os.makedirs('./model_weights')
if not os.path.exists('./logs'):
    os.makedirs('./logs') 

# training function
def train(net, optimizer, loss, train_loader, freeze_bn=False, swa=False):
    '''
    uses the data loader to grab a batch of images
    pushes images through network and gathers predictions
    updates network weights by evaluating the loss functions
    '''
    # set network to train mode
    #if args.gpu == 99:
    net.train()
    #else:
    #    net.train(mode=True, freeze_bn=args.freeze_bn)
    # keep track of our loss
    iter_loss = 0.

    t0 = time.time()
    ll = len(train_loader)
    # loop over the images for the desired amount
    for i, data in enumerate(train_loader):
        imgs = data[0].to(device)
        labels = data[1].to(device)

        # get predictions
        label_preds = net(imgs)
        #print(len(msk_preds), len(msks))
        # calculate loss
        tloss = loss(label_preds, labels)

        # zero gradients from previous run
        optimizer.zero_grad()
        #calculate gradients
        tloss.backward()
        # update weights
        optimizer.step()

        # get training stats
        iter_loss += tloss.item()

        # make a cool terminal output
        tc = time.time() - t0
        tr = int(tc*(ll-i-1)/(i+1))
        sys.stdout.write('\r')
        sys.stdout.write('B: {:>3}/{:<3} | Loss: {:.4} | ETA: {:d}s'.
            format(i+1, ll, tloss.item(), tr))

        if (i == 5 and args.dev_mode == True):
            print("\nDev mode on. Prematurely stopping epoch training.")
            break

    epoch_loss = iter_loss / (len(train_loader.dataset) / args.batch_size)
    print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

    return epoch_loss

# validation function
def valid(net, optimizer, loss, valid_loader, save_imgs=False, fold_num=0):
    net.eval() 
    #keep track of preds
    val_labels = torch.Tensor(len(valid_loader.dataset), 28)
    val_preds = torch.Tensor(len(valid_loader.dataset), 28)
    ci = 0

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
            sys.stdout.write('\r')
            sys.stdout.write('E: {:>3}/{:<3}'.format(i+1, 
                                                len(valid_loader)))
            
    epoch_vloss = loss(val_preds, val_labels)
    epoch_vf1 = macro_f1(val_preds>0, val_labels)
    epoch_vacc = accuracy(val_preds>0, val_labels)
    print('Avg Eval Loss: {:.4}, Avg Eval Macro F1: {:.4}, Avg Eval Acc. {:.4}'.
        format(epoch_vloss, epoch_vf1, epoch_vacc))
    return epoch_vloss, epoch_vf1

def train_network(net, fold=0, model_ckpt=None):
    # train the network, allow for keyboard interrupt
    try:
        # define optimizer
        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,net.parameters()), lr=0.00005)

        # get the loaders
        train_loader, valid_loader = get_data_loaders(imsize=args.imsize,
                                                      batch_size=args.batch_size)

        loss = FocalLoss()
        # loss = nn.BCEWithLogitsLoss()

        # training flags
        swa = False
        freeze_bn = False
        save_imgs = False
        train_losses = []
        valid_losses = []
        valid_ious = []

        valid_patience = 0
        best_val_metric = float('inf')
        cycle = 0
        swa_n = 0
        t_ = 0

        print('Training ...')
        print('Saving to ', model_ckpt)
        for e in range(args.epochs):
            print('\n' + 'Epoch {}/{}'.format(e, args.epochs))

            start = time.time()

            t_l = train(net, optimizer, loss, train_loader, freeze_bn)
            v_l, v_f1 = valid(net, optimizer, loss, valid_loader, save_imgs, fold)

            # save the model on best validation loss
            if v_l < best_val_metric:
                net.eval()
                torch.save(net.state_dict(), './model_weights/best_{}_{}.pth'.format(args.model_name,
                                                                                      args.exp_name))
                best_val_metric = v_l
                valid_patience = 0
                print('Best val metric achieved, model saved. metric = {}'.format(v_l))
            else:
                valid_patience += 1

            train_losses.append(t_l)
            valid_losses.append(v_l)

            t_ += 1
            print('Time: {}'.format(time.time()-start))

    except KeyboardInterrupt:
        pass

    net.eval()
    torch.save(net.state_dict(), './model_weights/swa_{}_{}.pth'.format(args.model_name, 
                                                                                 args.exp_name))

    return best_val_iou

def main_train():

    model_params = [args.model_name, args.exp_name]
    MODEL_CKPT = './model_weights/best_{}_{}.pth'.format(*model_params)

    # net = Atlas_DenseNet(model = args.model_name, bn_size=4, drop_rate=args.drop_rate)
    net = ResNet()
    
    net = nn.parallel.DataParallel(net)
    net.to(device)

    train_network(net, model_ckpt=MODEL_CKPT)

if __name__ == '__main__':
    main_train()



