import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import os


class ListLoader(data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list
        self.input_h = 224
        self.input_w = 224
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*self.mean_std)
        ])

    def _pre_process(self, raw_img):
        raw_img = Image.fromarray(raw_img)
        raw_img = self.transform(raw_img)
        return raw_img

    def __getitem__(self, idx):
        m_img = cv2.imread(self.path_list[idx])
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        m_img = cv2.resize(m_img, (224, 224), cv2.INTER_LINEAR)
        m_img = self._pre_process(m_img)
        return m_img

    def __len__(self):
        return len(self.path_list)


class ClassifierApi(object):
    def __init__(self, **kwargs):

        self.model_path = kwargs.get('model_path') or 'frozen_model.pt'
        self.device = kwargs.get('device') or None
        self.n_gpu = kwargs.get('n_gpu') or None
        self.gpu_list = list(range(self.n_gpu))


        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._construct_model(self.model_path)

        pass

    def _construct_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def _infer(self, x):
        with torch.no_grad():
            if self.n_gpu > 1:
                y = torch.nn.parallel.data_parallel(self.model, x, self.gpu_list)
            else:
                y = self.model(x)

            y = torch.nn.functional.softmax(y, dim=1)
            value, label = torch.max(y, 1)
            label = label.cpu().numpy()
            value = value.cpu().numpy()
            return value, label

    def exec(self, tensor):
        value, label = self._infer(tensor.to(self.device))
        return dict(score=[value, label])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    def test(epoch, project, dirname):
        my_inference = ClassifierApi(model_path='/defaultShare/share/wujl/83/master_models/resnext101_19mixcut/model_{}_frozen.pt'.format(epoch), n_gpu=1)
        img_dir = '/defaultShare/share/wujl/83/test/{}/croppic/{}'.format(project, dirname)
        txt_dir = '/defaultShare/share/wujl/83/test/{}/txtpath'.format(project)
        os.makedirs(txt_dir, exist_ok=True)
        print(img_dir)
        save_txt = os.path.join(txt_dir, '{}_{}.txt'.format(dirname, epoch))
        print(save_txt)
        txt_file = open(save_txt, 'w')
        name_list = os.listdir(img_dir)
        img_list = list()
        for name in name_list:
            img_list.append(os.path.join(img_dir, name))
        data_set = ListLoader(path_list=img_list)
        data_loader = DataLoader(data_set, batch_size=32, num_workers=16, shuffle=False)
        score_list = list()
        label_list = list()
        for index, img in enumerate(data_loader):
            batch_score = my_inference.exec(img)
            print(batch_score)
            scores, label = batch_score['score']
            score_list.extend(list(scores))
            label_list.extend(list(label))
            print(score_list)
            print('{}/{}'.format((index + 1) *len(list(batch_score['score'])), len(name_list)))
        print(np.sum(score_list))
        for index, name in enumerate(name_list):
            txt_file.write('{}, {}, {}\n'.format(name, score_list[index], label_list[index]))
        txt_file.close()
    epochs = [20]
    project = '9_21'
    dirnames = ['croppre_0.6', 'croprecall_0.6_0.3']
    for epoch in epochs:
        for dirname in dirnames:
            print('[*]! the epoch {} begin'.format(epoch))
            test(epoch, dirname=dirname, project=project)

