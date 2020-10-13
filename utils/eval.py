import torch 
from utils.process_bar import progress_bar
import numpy as np
from config.value_config import NUMCLASS, MAPLIST

def eval_recall(testloader, net, gpus):
    net.eval()
    total_num = 0 
    tp = 0
    prep = 0
    labelp = 0 
    acc = 0
    for i, data in enumerate(testloader):
        with torch.no_grad():
            if gpus>0:
                img, label = data[0].cuda(), data[1].cuda() 
            else:
                img, label = data[0], data[1]
            total_num += img.size(0)
            pre = net(img) 
            _, prelabel = torch.max(pre, 1)
            acc += torch.sum(prelabel.data == label.data)
            prep += torch.sum(prelabel.data==1)
            labelp += torch.sum(label.data==1)
            tp += torch.sum((prelabel+label).data == 2)
            progress_bar(i, len(testloader), 'val acc')
    test_acc = float(acc)/total_num 
    recall = float(tp.item())/(labelp.item() + 1e-6)
    precision = float(tp.item())/(prep.item()+1e-6)
    return test_acc, recall, precision

def eval_fuse(testloader, net, gpus):
    fuse_matrix = np.zeros((NUMCLASS, NUMCLASS))
    net.eval()
    total_num = 0
    acc = 0
    for i, data in enumerate(testloader):
        with torch.no_grad():
            if gpus>0:
                img, label = data[0].cuda(), data[1].cuda()
            else:
                img, label = data[0], data[1]
            total_num += img.size(0)
            pre = net(img)
            _, prelabel = torch.max(pre, 1)
            for i in range(NUMCLASS):
                for j in range(NUMCLASS):
                    fuse_matrix[i][j] += torch.sum((label.data==i)&(prelabel.data==j)).item()
            acc += torch.sum(prelabel.data == label.data)
            progress_bar(i, len(testloader), 'val acc')
    test_acc = float(acc)/total_num
    recalldic = {}
    precisiondic = {}
    for i in range(NUMCLASS):
        t = fuse_matrix[i][i]
        prenum = np.sum(fuse_matrix[:, i])
        num = np.sum(fuse_matrix[i,:])
        recalldic[i] = t/num
        precisiondic[i] = t/prenum
    print(fuse_matrix)

    return test_acc, recalldic, precisiondic

def displaymetric(recalldic, precisiondic):
    for key in recalldic.keys():
        name = MAPLIST[key]
        print('[*]! {} recall is : {}'.format(name, recalldic[key]))
    for key in precisiondic.keys():
        name = MAPLIST[key]
        print('[*]! {} precision is : {}'.format(name, precisiondic[key]))

