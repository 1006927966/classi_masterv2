"""
this is the DBSCAN FILTER, DBSCAN contains lots of parameters.for example
eps (领域半径)
min_samples(最小样本数)
return the core_samples(indices of point), labels（the label of every point）
this is a fail trick
"""

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from PIL import Image
import torchvision.transforms as T
import cv2
import torch
from utils.model_select import get_network
from utils.tsne import main



def pad_shot(img):
    w, h = img.size[-2:]
    s = max(w, h)
    new = Image.new('RGB', (s, s))
    new.paste(img, ((s-w)//2, (s-h)//2))
    return new


def transform_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = pad_shot(img)
    img = T.Resize(256, Image.BILINEAR)(img)
    img = T.CenterCrop((224, 224))(img)
    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = np.array([img.numpy()])
    img = torch.from_numpy(img)
    return img


class HookFeature:
    def __init__(self, dirPath, model):
        self.dirPath = dirPath
        self.names = [name.replace(' ', '') for name in os.listdir(self.dirPath)]
        self.model = model
        self.labels = [int(i.split('_')[0]) for i in self.names]
        self.vecs = []

    def change_label(self, tag):
        if tag == '非常淡':
            label = 0
        elif tag == '浓密正好':
            label = 2
        elif tag == '偏淡':
            label = 1
        elif tag == '偏浓':
            label = 3
        return label

    def hook_fn(self, module, input, output):
        self.vecs.append(output.data.cpu().numpy()[0])
        print('========feature========')

    def get_data(self, tag):
        for name, module in self.model.named_modules():
            if name == tag:
                module.register_forward_hook(self.hook_fn)
        file_names = os.listdir(self.dirPath)
        for file_name in file_names:
            file_path = os.path.join(self.dirPath, file_name)
            img = transform_img(file_path)
            self.model(img)
        return self.vecs, self.labels


def get_class_data(vecs, labels):
    labels = np.array(labels)
    label_set = np.unique(labels)
    datas_dic = {}
    for label in label_set:
        data_list = []
        indexes = np.where(labels==label)[0]
        for index in indexes:
            data_list.append(vecs[index])
        datas_dic[label] = data_list
    return datas_dic


def pca_demension(dim, x):
    pca = PCA(n_components=dim)
    newx = pca.fit_transform(x)
    return newx


def dbscan_cluster(x, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    return db.labels_


if __name__ == '__main__':
    dirpath = '/home/pc/gitcode/batch-dropblock-network/data/eyeData/bounding_box_test'
    state_path = '/home/pc/gitcode/multi_label/vgg16_multi_eye_shape_concetration/epoch139_test_acc0.5077519379844961.pth'
    net = get_network('vgg16', 5)
    tag = 'classifier.3'
    state_dic = torch.load(state_path, map_location='cpu')
    net.load_state_dict(state_dic)
    net = net.cpu()
    net.eval()
    hook_feature = HookFeature(dirpath, net)
    vecs, label = hook_feature.get_data(tag)
    class_dic = get_class_data(vecs, label)
    class0_data = class_dic[0]
    print(class0_data)
    print(np.shape(class0_data))
    newx = pca_demension(90, class0_data)
    print(np.shape(newx))
    labels = dbscan_cluster(newx, 0.01, 5)
    print(labels)
    print(np.unique(labels))
    main(newx, labels)




