""" helper function
author baiyu
"""

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_network(key, num_cls=2, use_gpu=False):
    """ return given network
    """

    if key == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn(num_cls)
    elif key == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(num_cls)
    elif key == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(num_cls)
    elif key == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn(num_cls)
    elif key == 'resnext':
        print('we will continue')
    elif key == 'efficientNetb0':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb1':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb2':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb3':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb4':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb5':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb6':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'efficientNetb7':
        from models.torchefficient import make_model
        net = make_model(key, num_cls)
    elif key == 'resnext50_32x8d':
        from models.resnext import make_model
        net = make_model(key)
    elif key == 'resnext101_32x8d':
        from models.resnext import make_model
        net = make_model(key)
    elif key == 'resnet50':
        from models.resnet import make_model
        net = make_model(key)
    elif key == 'resnet18':
        from models.resnet import make_model
        net = make_model(key)
    elif key == 'resnet34':
        from models.resnet import make_model
        net = make_model(key)
    elif key == 'resnet101':
        from models.resnet import make_model
        net = make_model(key)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    if use_gpu:
        net = net.cuda()
    return net