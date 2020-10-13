import sys
import os
curpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.split(curpath)[0]
sys.path.append(rootpath)
print(rootpath)

from config.value_config import *
import torch 
from utils.model_select import get_network
#from efficientnet_pytorch import EfficientNet
import torch.nn as nn


def frozen_model(dictpath):
    name = dictpath.split('/')[-1]
    epoch = name.split('_')[1]
    epoch = epoch.split('.')[0]
    model = get_network(MODELNAME, NUMCLASS)
    model.load_state_dict(torch.load(dictpath, map_location='cpu'))
    model.eval()
    model.cuda()
    example = torch.randn(1, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        trace_script_module = torch.jit.trace(model, example.cuda())
        trace_script_module.save(os.path.join(MODELSAVEPATH, 'model_{}_frozen.pt'.format(epoch)))
    print('[*]! the jit is create now')


#
# def frozen_efficinetdet(key, dictpath):
#     name = dictpath.split('/')[-1]
#     epoch = name.split('_')[0]
#     epoch = epoch.split('.')[0]
#     if key == "efficientNetb0":
#         name = "efficientnet-b0"
#     elif key == "efficientNetb1":
#         name = "efficientnet-b1"
#     elif key == "efficientNetb2":
#         name = "efficientnet-b2"
#     elif key == "efficientNetb3":
#         name = "efficientnet-b3"
#     elif key == "efficientNetb4":
#         name = "efficientnet-b4"
#     elif key == "efficientNetb5":
#         name = "efficientnet-b5"
#     elif key == "efficientNetb6":
#         name = "efficientnet-b6"
#     elif key == "efficientNetb7":
#         name = "efficientnet-b7"
#     model = EfficientNet.from_name(name)
#     model.set_swish(memory_efficient=False)
#     num_ftrs = model._fc.in_features
#     model._fc = nn.Linear(num_ftrs, NUMCLASS)
#     model.load_state_dict(torch.load(dictpath, map_location='cpu'))
#     model.eval()
#     model.cuda()
#     example = torch.randn(1, 3, HEIGHT, WIDTH)
#     with torch.no_grad():
#         trace_script_module = torch.jit.trace(model, example)
#         trace_script_module.save(os.path.join(MODELSAVEPATH, 'model_{}_frozen.pt'.format(epoch)))
#     print('[*]! the jit is create now')
#

if __name__ == '__main__':
    epochs = []
    for i in range(18, 59, 2):
        epochs.append(i)
    for epoch in epochs:
        dicpath = '/defaultShare/share/wujl/83/master_models/resnext101_19mixcut/model_{}.pth'.format(epoch)
        frozen_model(dicpath)

     
     
     
    

