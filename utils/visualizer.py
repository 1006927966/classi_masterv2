import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable


def get_lastlayer_params(net):
    """get last trainable layer of a net
    Args:
        network architectur

    Returns:
        last layer weights and last layer bias
    """
    last_layer_weights = None
    last_layer_bias = None
    for name, para in net.named_parameters():
        if 'weight' in name:
            last_layer_weights = para
        if 'bias' in name:
            last_layer_bias = para

    return last_layer_weights, last_layer_bias

def visualize_network(writer, net, width):
    """visualize network architecture"""
    input_tensor = torch.Tensor(3, 3, width, width)
    input_tensor = input_tensor.to(next(net.parameters()).device)
    writer.add_graph(net, Variable(input_tensor, requires_grad=True))


def visualize_lastlayer(writer, net, n_iter):
    """visualize last layer grads"""
    weights, bias = get_lastlayer_params(net)
    writer.add_scalar('LastLayerGradients/grad_norm2_weights', weights.grad.norm(), n_iter)
    writer.add_scalar('LastLayerGradients/grad_norm2_bias', bias.grad.norm(), n_iter)


def visualize_train_loss(writer, loss, n_iter):
    """visualize training loss"""
    writer.add_scalar('Train/loss', loss, n_iter)


def visualize_param_hist(writer, net, epoch):
    """visualize histogram of params"""
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def visualize_test_loss(writer, loss, epoch):
    """visualize test loss"""
    writer.add_scalar('Test/loss', loss, epoch)


def visualize_test_acc(writer, acc, epoch):
    """visualize test acc"""
    writer.add_scalar('Test/Accuracy', acc, epoch)


def visualize_learning_rate(writer, lr, epoch):
    """visualize learning rate, learning rate always like optimizer.param_groups[0]['lr']"""
    writer.add_scalar('Train/LearningRate', lr, epoch)


def init_weights(net):
    """the weights of conv layer and fully connected layers
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net
