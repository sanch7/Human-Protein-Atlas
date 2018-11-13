import os, sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.densenet import Atlas_DenseNet
from utils.dataloader import ProteinDataset

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--imsize', default=256, type=int, 
                    help='image size')
parser.add_argument('--batch_size', default=16, type=int, 
                    help='size of batches')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of epochs')
parser.add_argument('--lr', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--l2', default=1e-4, type=float,
                    help='l2 regularization for model')
parser.add_argument('--es_patience', default=50, type=int, 
                    help='early stopping patience')
parser.add_argument('--model_name', default='resunet', type=str,
                    help='name of model for saving/loading weights')
parser.add_argument('--exp_name', default='atlas_protein', type=str,
                    help='name of experiment for saving files')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # loop over the images for the desired amount
    for i, data in enumerate(train_loader):
        imgs = data['image'].to(device)
        labels = data['labels'] 

        # get predictions
        label_preds = net(labels)
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
        iter_loss += loss.item()
        # make a cool terminal output
        sys.stdout.write('\r')
        sys.stdout.write('B: {:>3}/{:<3} | {:.4}'.format(i+1, 
                                            len(train_loader),
                                            loss.item()))

    epoch_loss = iter_loss / (len(train_loader.dataset) / args.batch_size)
    print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

    return epoch_loss

# validation function
def valid(net, optimizer, loss, valid_loader, save_imgs=False, fold_num=0):
    net.eval() 
    # keep track of losses
    val_iter_loss = 0.
    # no gradients during validation
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            valid_imgs = data['image'].to(device)
            valid_labels = data['labels']
            
            # get predictions
            label_vpreds = net(valid_imgs)
       
            # calculate loss
            vloss = loss(label_vpreds, valid_labels)
 
            # get validation stats
            val_iter_loss += vloss.item()
            
    epoch_vloss = val_iter_loss / (len(valid_loader.dataset) / args.batch_size)
    print('Avg Eval Loss: {:.4}'.format(epoch_vloss))
    return epoch_vloss

def train_network(net, fold=0, model_ckpt=None):
    # train the network, allow for keyboard interrupt
    try:
        # define optimizer
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.l2)
        # get the loaders
        train_loader, valid_loader = get_data_loaders(imsize=args.imsize,
                                                      batch_size=args.batch_size)

        loss = torch.nn.MultiLabelSoftMarginLoss(weight=None, 
        	size_average=None, reduce=None, reduction='elementwise_mean')

        # training flags
        swa = False
        freeze_bn = False
        save_imgs = False
        train_losses = []
        valid_losses = []
        valid_ious = []

        valid_patience = 0
        best_val_metric = 1000.0
        best_val_iou = 0.0
        cycle = 0
        swa_n = 0
        t_ = 0

        print('Training ...')
        for e in range(args.epochs):
            print('\n' + 'Epoch {}/{}'.format(e, args.epochs))

            start = time.time()

            t_l = train(net, optimizer, loss, train_loader, freeze_bn)
            v_l = valid(net, optimizer, loss, valid_loader, save_imgs, fold)

            # save the model on best validation loss
            if v_l < best_val_loss:
                net.eval()
                torch.save(net.state_dict(), '../model_weights/best_{}_{}.pth'.format(args.model_name,
                                                                                      args.exp_name))
                best_val_metric = v_l
                valid_patience = 0
            else:
                valid_patience += 1

            train_losses.append(t_l)
            valid_losses.append(v_l)

            t_ += 1
            print('Time: {}'.format(time.time()-start))

    except KeyboardInterrupt:
        pass

    net.eval()
    torch.save(net.state_dict(), '../model_weights/swa_{}_{}.pth'.format(args.model_name, 
                                                                                 args.exp_name))

    return best_val_iou

def main_train():

    model_params = [args.model_name, args.exp_name]
    MODEL_CKPT = './model_weights/best_{}_{}.pth'.format(*model_params)

    net = Atlas_DenseNet(model = args.model_name, bn_size=4, drop_rate=0.)
    
    net = nn.parallel.DataParallel(net)
    net.to(device)

    train_network(net, model_ckpt=MODEL_CKPT)

if __name__ == '__main__':
	main_train()



