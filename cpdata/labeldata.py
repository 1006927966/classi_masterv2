import os
import shutil

class COPY:
    def __init__(self, name):
        self.maplist = ['badcase', 'feigang', 'hanzha', 'laji00', 'laji01', 'laji02', 'laji03', 'luanfang', 'zangwu',
           'zhufei', 'xianshu_luan', 'xianshu_zhengqi', 'xiaojian_luan', 'xiaojian_zhengqi', 'pip_luan', 'pip_zhengqi', 'zhixiang', 'bancai_luan', 'bancai_zhengqi']
        self.name = name
        self.picroot = '/defaultShare/share/wujl/83/online_data/cropdata'
        self.txtroot = '/defaultShare/share/wujl/83/online_data'
        self.saveroot = '/defaultShare/share/wujl/83/online_data/labeldata'
        self.txtpath = self.gettxtpath()
        self.picdir, self.savedir = self.getpicpath()

    def gettxtpath(self):
        return os.path.join(self.txtroot, self.name)

    def getpicpath(self):
        day = self.name.split('_')[0]
        picdir = os.path.join(self.picroot, day)
        savedir = os.path.join(self.saveroot, day)
        os.makedirs(savedir, exist_ok=True)
        return picdir, savedir

    def parsetxt(self):
        with open(self.txtpath, 'r') as f:
            lines = f.readlines()
        picnames = []
        labels = []
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            picnames.append(factors[0])
            label = self.maplist[int(factors[-1])]
            labels.append(label)
        return picnames, labels

    def copypic(self):
        picnames, labels = self.parsetxt()
        for i in range(len(picnames)):
            readfile = os.path.join(self.picdir, picnames[i])
            writedir = os.path.join(self.savedir, labels[i])
            os.makedirs(writedir, exist_ok=True)
            writefile = os.path.join(writedir, picnames[i])
            print(writefile)
            shutil.copy(readfile, writefile)


if __name__ == '__main__':
    names = ['2020-09-03_20.txt', '2020-09-05_20.txt', '2020-09-10_20.txt', '2020-09-12_20.txt']
    for name in names:
        copy = COPY(name)
        copy.copypic()
