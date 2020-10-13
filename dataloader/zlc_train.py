import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision
from config.value_config import *

class TrainData(data.Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.scale_center = 1.0

        if self.train:
            self.root = os.path.join(DATAPATH, 'train')
        else:
            self.root = os.path.join(DATAPATH, 'val')

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
        if self.train:
            self.data_path_0 = os.path.join(self.root, '0')
            self.data_path_1 = os.path.join(self.root, '1')
        else:
            self.data_path_0 = os.path.join(self.root, '0')
            self.data_path_1 = os.path.join(self.root, '1')
        img_list_0 = os.listdir(self.data_path_0)
        img_list_1 = os.listdir(self.data_path_1)

        self.img_list_0 = img_list_0
        self.img_list_1 = img_list_1


    def augment_resize(self, m_img, scale=0.2):

        seed = self.scale_center - scale * (self.scale_center - 2.0 * self.scale_center * random.random())
        w, h = m_img.size
        w, h = int(w * seed), int(h * seed)
        m_img = cv2.resize(np.array(m_img), (w, h), cv2.INTER_LINEAR)
        m_img = Image.fromarray(m_img)
        return m_img

    def __getitem__(self, idx):

        if random.random() < 0.5:
            img_name = os.path.join(self.data_path_0, self.img_list_0[random.randint(0, len(self.img_list_0) - 1)])
            target = 0
        else:
            img_name = os.path.join(self.data_path_1, self.img_list_1[random.randint(0, len(self.img_list_1) - 1)])
            target = 1

        img = Image.open(img_name).convert('RGB')
        img = self.augment_resize(img)
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        img = cv2.resize(img, (self.input_w, self.input_h), cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img = self.input_transform(img)
        return img, target

    def __len__(self):
        return len(self.img_list_0) + len(self.img_list_1)

