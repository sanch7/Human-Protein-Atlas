import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

def accuracy(preds, target):
    return np.mean((preds > 0.) == target)

def macro_f1(preds, target, class_wise=False):
    score = []
    for i in range(preds.shape[1]):
        score.append(f1_score(target[:, i], preds[:, i]))
    if class_wise:
        return np.array(score)
    else:
        return np.mean(score)

def f1_score_slow(y_true, y_pred, threshold=0.5):
    """
    Usage: f1_score(py_true, py_pred)
    """
    return fbeta_score(y_true, y_pred, 1, threshold)


def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean((precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        tp = (target*input).sum(0)
        # tn = ((1-target)*(1-input)).sum(0)
        fp = ((1-target)*input).sum(0)
        fn = (target*(1-input)).sum(0)

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)

        f1 = 2*p*r / (p+r+1e-9)
        f1[f1!=f1] = 0.
        return 1 - f1.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2.*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalTverskyLoss(nn.Module):
    """
    https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    Focal Tversky Loss. Tversky loss is a special case with gamma = 1
    """
    def __init__(self, alpha = 0.4, gamma = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def tversky(self, input, target):
        smooth = 1.
        input = torch.sigmoid(input)

        target_pos = target.view(-1)
        input_pos = input.view(-1)
        true_pos = (target_pos * input_pos).sum()
        false_neg = (target_pos * (1-input_pos)).sum()
        false_pos = ((1-target_pos)*input_pos).sum()
        return (true_pos + smooth)/(true_pos + self.alpha*false_neg + \
                        (1-self.alpha)*false_pos + smooth)

    def forward(self, input, target):
        pt_1 = self.tversky(input, target)
        return (1-pt_1).pow(self.gamma)