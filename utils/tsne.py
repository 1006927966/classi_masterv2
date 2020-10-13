"t-SNE 对手写数字进行可视化"""
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


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i]+1) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main(data, label):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    plt.show(fig)


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
    print(vecs)
    print(label)
    main(vecs, label)



