import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr*self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def adjust_learning_rate_linear(optimizer, epoch, initial_lr):
    lr = initial_lr *(0.9**(epoch//1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_traingle(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
    valid_epoch = epoch %8
    k = (max_lr - min_lr)/(cycle//2)
    lr = max_lr - abs(valid_epoch - cycle//2)*k
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_warmup(optimizer, epoch, max_lr, min_lr=0.0001, cycle=8):
    valid_epoch = epoch % cycle
    valid_max_lr = max_lr * (0.7**(epoch // cycle))
    delta_lr = (valid_max_lr - min_lr)
    k = delta_lr/cycle
    if epoch < cycle:
        if epoch<=cycle//2:
            lr = min_lr + 0.5*delta_lr + valid_epoch * k
        else:
            lr = max_lr - (epoch - cycle//2)*2*k
    else:
        lr = valid_max_lr - (valid_epoch - cycle)*k
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr_scheduler(optimizer, tag):
    if  tag == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)
    elif tag == 'MultiStep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.1, last_epoch= -1)
    elif tag == 'ExLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif tag == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min= 0.00001, last_epoch=-1)
    return scheduler

