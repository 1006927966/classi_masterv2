import os
import torch.utils.data as data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from utils.multi_label import MultiOutModel
from utils.process_bar import progress_bar
from tensorboardX import SummaryWriter
from dataloader.select_loader import loader 
from config.value_config import *
from utils.model_select import  get_network
import torch
from utils.lr_scheduler import WarmUpLR, get_lr_scheduler
from utils.eval import eval_fuse, displaymetric
import time 
from utils.mixup import mixup_data, mixup_criterion

os.environ['CUDA_VISIBLE_DEVICES'] = GPUS
os.makedirs(LOGPATH, exist_ok=True)
summary_writer = SummaryWriter(LOGPATH)
os.makedirs(MODELSAVEPATH, exist_ok=True)


trainloader, testloader = loader(LOADERNAME)
beg = time.time()
net = get_network(MODELNAME, NUMCLASS)
if LOADING:
    net.load_state_dict(torch.load(LOADPATH, map_location='cpu'))
net.cuda() 
if torch.cuda.device_count()>1:
    net = DataParallel(net)
end = time.time()
print('[*]! model load time is{}'.format(end-beg))
iters = len(trainloader)
optimizer = torch.optim.SGD(net.parameters(), lr=INITLR, momentum=0.9, weight_decay=WD)
scheduler = get_lr_scheduler(optimizer, LRTAG)
warmup = WarmUpLR(optimizer, iters*WARM)

print('[*] train start !!!!!!!!!!!')
for epoch in range(EPOCHS):
    net.train()
    train_loss = 0
    total = 0
    best_acc = 0
    best_epoch = 0
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer.zero_grad()
        if MIXUP:
            img, labela, labelb, lam = mixup_data(img, label)
            pre = net(img)
            criterion = torch.nn.CrossEntropyLoss()
            loss = mixup_criterion(criterion, pre, labela, labelb, lam)
        else:
            pre = net(img)
            loss = torch.nn.CrossEntropyLoss()(pre, label)
        train_loss += loss * batch_size
        total += batch_size
        loss.backward()
        optimizer.step()
        progress_bar(i, len(trainloader), 'train')
    if epoch > WARM:
        scheduler.step()
    else:
        warmup.step()
    print('[*] epoch:{} - train loss: {:.3f}'.format(epoch, train_loss/total))
    acc, recalldic, precisiondic = eval_fuse(testloader, net, torch.cuda.device_count())
    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
    print('[*] epoch:{} - test acc: {:.3f} - best acc: {}_{}'.format(epoch, acc, best_epoch, best_acc))
    displaymetric(recalldic, precisiondic)
# save train_loss and val_acc
    if epoch%SAVEFREQ == 0:
        summary_writer.add_scalar('loss', train_loss.item(), epoch)
        summary_writer.add_scalar('val_acc', acc, epoch)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        os.makedirs(MODELSAVEPATH, exist_ok=True)
        if torch.cuda.device_count() > 1:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        torch.save(net_state_dict, os.path.join(MODELSAVEPATH, 'model_{}.pth'.format(epoch)))
        print('[*] change the best model')
print('[*] training finished')


