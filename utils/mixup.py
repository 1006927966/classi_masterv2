import numpy as np
import torch

"""
this section is about mix-up, it is used in a batch, shuffle the order. 
calculate the mix-up between the shuffle batch and origin batch.

this tricks is used as loss function at the end of the model. just like this 

the first inputs represent the data from the dataloader。

the criterion represent the cross-entripy()

inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha)
output = model(inputs)
loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)

"""


def mixup_data(x, y, alpha=0.2):
    """Returns mixed up inputs pairs of targets and lambda"""
    if alpha > 0:
# 一种随机抽样的算法方式。
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
# 随机打乱一个数字序列
    index = torch.randperm(batch_size)
    index = index.to(x.device)

    lam = max(lam, 1 - lam)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)








