import os, sys
import time
import argparse
import json
import pprint
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import models.model_list as model_list
from models.wide_resnet_cifar_attention import WideResNetAttention
from utils.dataloader import get_data_loaders
from utils.metrics import FocalLoss, DiceLoss, F1Loss, FocalTverskyLoss, accuracy, macro_f1
from utils.misc import log_metrics, cosine_annealing_lr

from evaluations import generate_preds, generate_submission
from make_submission import main_subm

parser = argparse.ArgumentParser(description='Atlas Protein')
parser.add_argument('--config', default='./configs/config.json', 
                    help="run configuration")
parser.add_argument('--dev_mode', action='store_true', default=False,
                    help='train only few batches per epoch')
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume training from the checkpoint')
parser.add_argument('--submission', action='store_true', default=False,
                    help='Generate submission')

# Model options
parser.add_argument('--depth', default=34, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dropout', type=float, default=0)

# Attention
parser.add_argument("--attention_depth", default=3, type=int, help="Painless attention depth")
parser.add_argument("--attention_width", default=1, type=int, help="Painless attention width")
parser.add_argument("--attention_type", default="softmax", type=str, help="How to compute attention masks")
parser.add_argument("--reg_w", default=0.001, type=float, help="Inter-mask regularization weight")
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

if not os.path.exists('./model_weights'):
    os.makedirs('./model_weights')
if not os.path.exists('./logs'):
    os.makedirs('./logs') 

# resume training
def load_checkpoint(model, optimizer, model_ckpt):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(model_ckpt):
        print("Resuming from checkpoint '{}'".format(model_ckpt))
        checkpoint = torch.load(model_ckpt)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['best_val_loss']
        print("Loaded checkpoint '{}' (epoch {})"
                  .format(model_ckpt, checkpoint['epoch']))

        model = model.to(device)
        model.train()
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    else:
        print("=> no checkpoint found at '{}'".format(model_ckpt))

    return model, optimizer, start_epoch, best_val_loss

# training function
def train(net, optimizer, loss, train_loader, freeze_bn=False):
    '''
    uses the data loader to grab a batch of images
    pushes images through network and gathers predictions
    updates network weights by evaluating the loss functions
    '''
    # set network to train mode
    net.train()

    # keep track of our loss
    iter_loss = 0.

    t0 = time.time()
    ll = len(train_loader)
    # loop over the images for the desired amount
    for i, data in enumerate(train_loader):
        imgs = data[0].to(device)
        labels = data[1].to(device)

        # get predictions
        label_preds, rloss = net(imgs)
        label_preds += 3.5
        #print(len(msk_preds), len(msks))
        # calculate loss
        tloss = loss(label_preds, labels)

        # Attention Reg Loss
        tloss += rloss

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
        sys.stdout.write('B: {:>3}/{:<3} | Loss: {:<7.4f} | ETA: {:>4d}s'.
            format(i+1, ll, tloss.item(), tr))

        if (i == 5 and args.dev_mode == True):
            print("\nDev mode on. Prematurely stopping epoch training.")
            break

    epoch_loss = iter_loss / (len(train_loader.dataset) / config.batch_size)
    print('\n' + 'Avg Train Loss: {:.4}'.format(epoch_loss))

    return epoch_loss

# validation function
def valid(net, optimizer, loss, valid_loader, save_imgs=False, fold_num=0):
    net.eval() 
    #keep track of preds
    val_preds, val_labels = generate_preds(net, valid_loader, attn=True)
    
    epoch_vloss = loss(val_preds, val_labels)
    epoch_vf1 = macro_f1(val_preds.numpy()>0., val_labels.numpy())
    epoch_vacc = accuracy(val_preds.numpy()>0., val_labels.numpy())
    print('Avg Eval Loss: {:.4}, Avg Eval Macro F1: {:.4}, Avg Eval Acc. {:.4}'.
        format(epoch_vloss, epoch_vf1, epoch_vacc))
    return epoch_vloss, epoch_vf1

def train_network(net, model_ckpt, fold=0):
    # train the network, allow for keyboard interrupt
    try:
        # define optimizer
        # optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9, weight_decay=configs.l2)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,net.parameters()), 
                            lr=config.lr)

        valid_patience = 0
        best_val_loss = None
        best_val_f1 = None
        cycle = 0
        t_ = 0

        if args.resume:
            net, optimizer, start_epoch, best_val_loss = load_checkpoint(net, 
                                        optimizer, model_ckpt)

        if config.reduce_lr_plateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', config.lr_scale,
                            config.lr_patience, True)

        if config.cosine_annealing:
            cos_lr, cycle_ends = cosine_annealing_lr(config.min_lr, config.max_lr, 
                    config.cycle_size, config.epochs, config.cycle_size_inc)

        # get the loaders
        train_loader, valid_loader = get_data_loaders(imsize=config.imsize,
                                                      num_channels=config.num_channels,
                                                      batch_size=config.batch_size,
                                                      test_size=config.test_size,
                                                      num_workers=config.num_workers,
                                                      preload=config.preload_data,
                                                      external_data=config.external_data)

        # loss = F1Loss()
        if hasattr(config, 'focal_gamma'):
            loss = FocalLoss(config.focal_gamma)
        else:
            loss = FocalLoss()
        # loss = nn.BCEWithLogitsLoss().cuda()
        # if hasattr(config, 'focal_gamma'):
        #     loss = FocalTverskyLoss(gamma = config.focal_gamma)
        # else:
        #     loss = FocalTverskyLoss()
        
        # training flags
        freeze_bn = False
        save_imgs = False
        train_losses = []
        valid_losses = []
        valid_f1s = []
        lr_hist = []

        print('Training ...')
        print('Saving to ', model_ckpt)
        for e in range(config.epochs):
            print('\n' + 'Epoch {}/{}'.format(e, config.epochs))

            start = time.time()

            t_l = train(net, optimizer, loss, train_loader, freeze_bn)
            v_l, v_f1 = valid(net, optimizer, loss, valid_loader, save_imgs, fold)

            if config.reduce_lr_plateau:
                scheduler.step(v_l)

            if config.cosine_annealing:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cos_lr[e]
                if (e in cycle_ends):
                    cycle = np.where(cycle_ends==e)[0][0]+1
                    net.eval()
                    torch.save(net.state_dict(), 
                        model_ckpt.replace('best', 'cycle{}'.format(cycle)))
                    print("Cycle {} completed. Saving model to {}".format(cycle, 
                        model_ckpt.replace('best', 'cycle{}'.format(cycle))))

            lr_hist.append(optimizer.param_groups[0]['lr'])

            state = {
                    'epoch': e,
                    'arch': config.model_name,
                    'state_dict': net.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer' : optimizer.state_dict(),
                }
            
            # save the model on best validation loss
            if best_val_loss is None or v_l < best_val_loss:
                best_val_loss = v_l
                
                net.eval()

                torch.save(state, model_ckpt)
                valid_patience = 0
                print('Best val loss achieved. loss = {:.4f}.'.
                    format(v_l), " Saving model to ", model_ckpt)

            # save the model on best validation f1
            # if best_val_f1 is None or v_f1 > best_val_f1:
            #     net.eval()
            #     torch.save(net.state_dict(), model_ckpt.replace('best', 'bestf1'))
            #     best_val_f1 = v_f1
            #     valid_patience = 0
            #     print('Best val F1 achieved. F1 = {:.4f}.'.
            #         format(v_f1), " Saving model to ", model_ckpt.replace('best', 'bestf1'))

                # if (e > 5):
                #     SUBM_OUT = './subm/{}_{}_epoch{}.csv'.format(
                #                     config.model_name, config.exp_name, str(e))
                #     generate_submission(net, config, SUBM_OUT)

            else:
                valid_patience += 1

            torch.save(state, model_ckpt.replace('best', 'latest'))

            train_losses.append(t_l)
            valid_losses.append(v_l)
            valid_f1s.append(v_f1)

            log_metrics(train_losses, valid_losses, valid_f1s, lr_hist, e, 
                            model_ckpt, config)

            t_ += 1
            print('Time: {:d}s'.format(int(time.time()-start)))

    except KeyboardInterrupt:
        pass

    gen_sub = input("\n\nGenerate submission while the GPU is still hot from training? [Y/n]: ")
    if gen_sub in ['Y', 'y', 'Yes', 'yes']:
        generate_submission(net, config)

def main_train():

    model_params = [config.model_name, config.exp_name]
    MODEL_CKPT = './model_weights/best_{}_{}.pth'.format(*model_params)

    # net = Atlas_DenseNet(model = config.model_name, bn_size=4, drop_rate=config.drop_rate)
    # Net = getattr(model_list, config.model_name)
    
    # net = Net(pretrained=config.pretrained, drop_rate=config.drop_rate, 
    #                 num_channels = config.num_channels)

    # net = nn.parallel.DataParallel(net)
    net = WideResNetAttention(args.depth, args.width, 28, args.dropout, 
                                args.attention_depth, args.attention_width, 
                                args.reg_w, args.attention_type)

    net.to(device)

    if args.submission:
        main_subm(net, config, attn=True)
        sys.exit()

    train_network(net, model_ckpt=MODEL_CKPT)

if __name__ == '__main__':
    main_train()



