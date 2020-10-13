import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch
import torch.nn as nn
import os
import numpy as np
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def getinput(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    return img

def getmodel(dicpath):
    model = models.__dict__['resnet50'](num_classes=128)
    dics = torch.load(dicpath, map_location='cpu')
    state_dict = dics['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

def infer(path, model):
    img = getinput(path)
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        latentcode = model(img)
    latentcode = nn.functional.normalize(latentcode, dim=1)
    if torch.cuda.is_available():
        latentcode = latentcode.cpu().numpy()
    else:
        latentcode = latentcode.numpy()
    return latentcode[0]

def cosin(target, selectboxes, model, topk, save=True, savepath=''):
    targetlat = infer(target, model)
    selects = os.listdir(selectboxes)
    values = []
    names = []
    for select in selects:
        selectpath = os.path.join(selectboxes, select)
        selectlat = infer(selectpath, model)
        value = np.dot(targetlat, selectlat)
        print(value)
        values.append(value)
        names.append(select)
    argindex = np.argsort(-np.array(values))
    print(argindex)
    print('*'*20)
    useful = []
    count = 0
    for index in argindex[:topk]:
        useful.append(names[index])
        if 'jpg' in names[index]:
            ip = names[index][:-4]
        if 'jpeg' in names[index]:
            ip = names[index][:-5]
        count += 1
        if save == True:
            readfile = os.path.join(selectboxes, names[index])
            writefile = os.path.join(savepath, str(count)+'.jpg')
            shutil.copy(readfile, writefile)

if __name__ == '__main__':
    classis = ['xianshuluan']
    dictpath = '/defaultShare/share/wujl/moco/featuremodels/checkpoint_0199.pth.tar'
    model = getmodel(dictpath)

    picroot = '/defaultShare/share/wujl/83/search/'
    saveroot = picroot + 'result'
    srcbox = picroot + 'ku'
    print(srcbox)
    for classi in classis:
        targetsrc = picroot + '{}'.format(classi)
        names = os.listdir(targetsrc)
        for name in names:
            targetpath = os.path.join(targetsrc, name)
            savepath = os.path.join(saveroot, classi, name[:-4])
            os.makedirs(savepath, exist_ok=True)
            cosin(targetpath, srcbox, model, 10, savepath=savepath)











