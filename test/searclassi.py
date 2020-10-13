import os
import shutil

class usefulpic:
    def __init__(self, project, epoch, rootdir, iou, label):
        self.project = project
        self.epoch = epoch
        self.rootdir = rootdir
        self.iou = iou
        self.label = label
    def savepic(self, name):
        origndir = os.path.join(self.rootdir, self.project, 'croppic', 'croppre')
        readfile = os.path.join(origndir, name)
        os.makedirs( os.path.join(self.rootdir, self.project, str(self.label)), exist_ok=True)
        writefile = os.path.join(self.rootdir, self.project, str(self.label), name)
        shutil.copy(readfile, writefile)


    def getclassi(self):
        usefulpic = []
        clssitxtdir = os.path.join(self.rootdir, self.project, 'txtpath')
        classitxtpath = os.path.join(clssitxtdir, 'croppre_{}.txt'.format(self.epoch))
        with open(classitxtpath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            classi = line.split(',')[-1]
            if int(classi) == self.label:
                usefulpic.append(line.split(',')[0])
        return usefulpic

    def getoverlap(self):
        txtdir = os.path.join(self.rootdir, self.project, 'txtpath')
        txtpath = os.path.join(txtdir, 'recall_{}.txt'.format(self.iou))
        usefulpic = self.getclassi()
        with open(txtpath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            name = line.split(',')[0]
            if name in usefulpic:
                self.savepic(name)

if __name__ == '__main__':
    project = '725_84'
    epoch = '18'
    rootdir = '/defaultShare/share/wujl/83/test'
    iou = 0.3
    label = 3
    use = usefulpic(project, epoch, rootdir, iou, label)
    use.getoverlap()