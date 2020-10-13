import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
from config.value_config import *
from utils.cutout import CutOut

class TrainData(data.Dataset):
    def __init__(self,  transform=None):
        self.scale_center = 1.0
        self.root = os.path.join(DATAPATH, 'train')
        self.input_h = HEIGHT
        self.input_w = WIDTH
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(180, expand=True),
                torchvision.transforms.RandomGrayscale(p=0.1)
                # torchvision.transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(0, 1))
            ])
        self.transform = transform
        if Cut == False:
            self.input_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*self.mean_std)
            ])
        else:
            self.input_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*self.mean_std),
                CutOut(1, 16)
            ])
        self.img_list = []
        for i in range(NUMCLASS):
            subdir = os.path.join(self.root, str(i))
            sublist = []
            names = os.listdir(subdir)
            for name in names:
                sublist.append(os.path.join(subdir, name))
            self.img_list.append(sublist)


    def augment_resize(self, m_img, scale=0.2):

        seed = self.scale_center - scale * (self.scale_center - 2.0 * self.scale_center * random.random())
        w, h = m_img.size
        w, h = int(w * seed), int(h * seed)
        m_img = cv2.resize(np.array(m_img), (w, h), cv2.INTER_LINEAR)
        m_img = Image.fromarray(m_img)
        return m_img

    def __getitem__(self, idx):
        classi = random.randint(0, NUMCLASS-1)
        target = classi
        sublist = self.img_list[classi]
        img_name = sublist[random.randint(0, len(sublist) -1)]
        img = Image.open(img_name).convert('RGB')
        img = self.augment_resize(img)
        if self.transform is not None:
         #   img = torchvision.transforms.RandomCrop((int(img.size[1]/1.5), int(img.size[0]/1.5)))(img)
            img = self.transform(img)
        img = np.array(img)
        img = cv2.resize(img, (self.input_h, self.input_w), cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = self.input_transform(img)
        return img, target

    def __len__(self):
        num = 0
        for sublist in self.img_list:
            num += len(sublist)
        return num

