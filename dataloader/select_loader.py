import torch.utils.data as data
from config.value_config import *
import time

def loader(key):
    if key == 'zlc':
        from .zlc_train import TrainData
        from .zlc_test import TestData
        trainDataSet = TrainData(train=False)
        testDataSet = TestData()
        trainloader = data.DataLoader(trainDataSet, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD, drop_last=False)
        testloader = data.DataLoader(testDataSet, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD, drop_last=False)
    elif key == 'zlc_multi':
        from .zlc_multi_train import TrainData
        from .zlc_multi_test import TestData
        traindataset = TrainData()
        testdataset = TestData()
        trainloader = data.DataLoader(traindataset, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD, drop_last=False)
        testloader = data.DataLoader(testdataset, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD, drop_last=False)
    elif key == 'zlc_multi_balance':
        from .zlc_balance_multi_train import TrainData
        from .zlc_multi_test import TestData
        traindataset = TrainData()
        testdataset = TestData()
        trainloader = data.DataLoader(traindataset, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD,
                                      drop_last=False)
        testloader = data.DataLoader(testdataset, batch_size=BATCHSIZE, shuffle=True, num_workers=THREAD,
                                     drop_last=False)
    else:
        print('[*]! the key error of dataloader!!!!')
    return trainloader, testloader







