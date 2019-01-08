import argparse

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def label_gen_tensor(labelstr):
    label = torch.zeros(28)
    labelstr = labelstr.split()
    for l in labelstr:
        label[int(l)]=1
    return label

def label_gen_np(labelstr):
    label = np.zeros(28, dtype='uint8')
    labelstr = labelstr.split()
    for l in labelstr:
        label[int(l)]=1
    return label

def save_pred(pred, th=0., SUBM_OUT='./subm/submission.csv', fill_empty=True):
    pred_list = []
    for line in pred:
        line -= th             # accomodate both class_wise and overall thresholding
        s = ' '.join(list([str(i) for i in np.nonzero(line>=0.)[0]]))
        if fill_empty and s == '':
            s = str(line.argmax())
        pred_list.append(s)
    
    sample_df = pd.read_csv('./data/sample_submission.csv')
    sample_df['Predicted'] = pred_list
    sample_df.to_csv(SUBM_OUT, header=True, index=False)
    print('Saved to ', SUBM_OUT)

def log_metrics(train_losses, valid_losses, valid_f1s, lr_hist, e, model_ckpt, config):
    _, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].plot(train_losses)
    axes[0, 1].plot(valid_losses)
    axes[1, 0].plot(valid_f1s)
    axes[1, 1].plot(lr_hist)
    axes[0, 0].set_title('Train Loss')
    axes[0, 1].set_title('Val Loss')
    axes[1, 0].set_title('Val F1')
    axes[1, 1].set_title('LR History')
    plt.suptitle("At Epoch {}, desc: {}".format(e+1, config.desc), fontsize=16)
    plt.savefig(model_ckpt.replace('model_weights', 'logs').replace('.pth', '.png'))

def cosine_annealing_lr(min_lr, max_lr, cycle_size, epochs, cycle_size_inc = 0):
    new_epochs = cycle_size
    n_cycles = 1
    temp_cs = cycle_size
    while (new_epochs <= epochs-temp_cs):
        temp_cs += cycle_size_inc
        new_epochs += temp_cs
        n_cycles += 1
    print("Performing {} epochs for {} cycles".format(new_epochs, n_cycles))
    
    cycle_e = 0
    lr = []
    cycle_ends = [0]
    for e in range(new_epochs):
        lr.append(min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos(cycle_e*np.pi/cycle_size)))
        cycle_e += 1
        if cycle_e == cycle_size:
            cycle_ends.append(cycle_e + cycle_ends[-1])
            cycle_e = 0
            cycle_size += cycle_size_inc
    cycle_ends = np.array(cycle_ends[1:]) - 1
    return lr, cycle_ends

def f1_loss_keras(y_true, y_pred):
    from keras import backend as K
    import tensorflow as tf
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def f1_keras(y_true, y_pred, THRESHOLD = 0.05):
    from keras import backend as K
    import tensorflow as tf

    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)