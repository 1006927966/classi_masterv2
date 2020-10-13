import os
import cv2
import torchvision
from PIL import Image
from torch.utils import data
from config.value_config import *

class TestData(data.Dataset):
    def __init__(self, transform=None):
        self.root = os.path.join(DATAPATH, 'val')
        self.input_h = HEIGHT
        self.input_w = WIDTH
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if transform is None:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(*self.mean_std)
            ])
        self.transform = transform
        self.img_list = []
        subdirs = os.listdir(self.root)
        for subdir in subdirs:
            subpath = os.path.join(self.root, subdir)
            names = os.listdir(subpath)
            for name in names:
                self.img_list.append(os.path.join(subpath, name))

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        target = int(img_name.split('/')[-2])
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_w, self.input_h), cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.img_list)
