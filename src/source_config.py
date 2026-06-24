import os
import sys
import torch
import traceback
import torch.nn as nn
import numpy as np


def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.log = open(filename, 'w')
        self.hook = sys.excepthook
        sys.excepthook = self.kill

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def kill(self, ttype, tvalue, ttraceback):
        for trace in traceback.format_exception(ttype, tvalue, ttraceback):
            print(trace)
        os.remove(self.filename)

    def flush(self):
        pass


def GDL_loss(pred, label):
    smooth = 1.
    weight = 1. - torch.sum(label, dim=(0, 2, 3)) / torch.sum(label)
    # weight = 1./(torch.sum(label, dim=(0, 2, 3))**2 + smooth); smooth = 0
    intersection = pred * label
    intersection = weight * torch.sum(intersection, dim=(0, 2, 3))
    intersection = torch.sum(intersection)
    union = pred + label
    union = weight * torch.sum(union, dim=(0, 2, 3))
    union = torch.sum(union)
    score = 1. - (2. * (intersection + smooth) / (union + smooth))
    return score


# def dice_coeff(pred, label, weight=None):
#     smooth = 1.
#     bs = pred.size(0)
#     m1 = pred.contiguous().view(bs, -1)
#     m2 = label.contiguous().view(bs, -1)
#     intersection = (m1 * m2).sum()
#     score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
#     return score

def dice_coeff(pred, label, weight=None):
    smooth = 1.
    bs = pred.size(0)
    m1 = pred.contiguous().view(bs, -1)
    m2 = label.contiguous().view(bs, -1)
    if weight is None:
        intersection = (m1 * m2).sum()
        score = 1 - ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth))
    else:
        w = weight.contiguous().view(bs, -1)
        intersection = (m1 * m2 * w).sum()
        score = 1 - ((2. * intersection + smooth) / ((m1*w).sum() + (m2*w).sum() + smooth))
    return score

# def dice_coeff(pred, label, weight=None):
#     smooth = 1.
#     bs = pred.size(0)
#     # 展平预测和标签张量
#     m1 = pred.contiguous().view(bs, -1)
#     m2 = label.contiguous().view(bs, -1)
    
#     # 处理权重张量
#     if weight is not None:
#         w = weight.contiguous().view(bs, -1)
#     else:
#         w = torch.ones_like(m1)  # 无权重时全1
    
#     # 计算加权交集和加权分母
#     intersection = (m1 * m2 * w).sum()
#     sum_m1 = (m1 * w).sum()
#     sum_m2 = (m2 * w).sum()
    
#     # 计算Dice系数和损失
#     dice = (2. * intersection + smooth) / (sum_m1 + sum_m2 + smooth)
#     score = 1 - dice
    
#     return score

def jaccard_loss(pred, label):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m1 = torch.abs(1 - m1)
    m2 = label.view(num, -1)  # Flatten
    m2 = torch.abs(1 - m2)
    score = 1 - ((torch.min(m1, m2).sum() + smooth) / (torch.max(m1, m2).sum() + smooth))
    return score


def p2p_loss(pred, label):
    # Mean Absolute Error (MAE)
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = label.view(num, -1)  # Flatten
    score = torch.mean(torch.abs(m2 - m1))
    return score


def bce_loss(pred, label, weight=None):
    if weight is None:
        score = torch.nn.BCELoss()(pred, label)
    else:
        score = (torch.nn.BCELoss(reduction='none')(pred, label) * weight).mean()
    return score


class Seg_loss(nn.Module):
    def __init__(self, lossmap):
        super(Seg_loss, self).__init__()
        self.tasks = lossmap
        self.tasks = tuple(sorted(self.tasks))
        self.lossmap = {
            'dice': eval('dice_coeff'),
            'bce': eval('bce_loss'),
            'jaccard': eval('jaccard_loss'),
            'p2p': eval('p2p_loss'),
            'gdl': eval('GDL_loss'),
        }

    def forward(self, logit_pred, label, weight=None):
        pred = torch.sigmoid(logit_pred)

        score = 0
        for task in self.tasks:
            if task in self.lossmap.keys():
                if weight is None:
                    score += self.lossmap[task](pred=pred, label=label)
                else:
                    score += self.lossmap[task](pred=pred, label=label, weight=weight)
        return score


class EpochLR(torch.optim.lr_scheduler._LRScheduler):
    # lr_n = lr_0 * (1 - epoch / epoch_nums)^gamma
    def __init__(self, optimizer, epochs, gamma=0.9, last_epoch=-1):
        self.lr = optimizer.param_groups[0]['lr']
        self.epochs = epochs
        self.gamma = gamma
        super(EpochLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.lr * pow((1. - self.last_epoch / self.epochs), self.gamma)]

