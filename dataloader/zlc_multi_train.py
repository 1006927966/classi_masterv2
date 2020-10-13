import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
from config.value_config import *

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
                torchvision.transforms.RandomGrayscale(p=0.1),
                # torchvision.transforms.RandomAffine(degrees=180, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(0, 1))
            ])
        self.transform = transform
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*self.mean_std)
        ])
        self.img_list = []
        subdirs = os.listdir(self.root)
        for subdir in subdirs:
            subpath = os.path.join(self.root, subdir)
            names = os.listdir(subpath)
            for name in names:
                self.img_list.append(os.path.join(subpath, name))

    def augment_resize(self, m_img, scale=0.2):

        seed = self.scale_center - scale * (self.scale_center - 2.0 * self.scale_center * random.random())
        w, h = m_img.size
        w, h = int(w * seed), int(h * seed)
        m_img = cv2.resize(np.array(m_img), (w, h), cv2.INTER_LINEAR)
        m_img = Image.fromarray(m_img)
        return m_img

    def __getitem__(self, idx):

        img_name = self.img_list[idx]
        target = int(img_name.split('/')[-2])
        img = Image.open(img_name).convert('RGB')
        img = self.augment_resize(img)
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        img = cv2.resize(img, (self.input_h, self.input_w), cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = self.input_transform(img)
        return img, target

    def __len__(self):
        return len(self.img_list)

