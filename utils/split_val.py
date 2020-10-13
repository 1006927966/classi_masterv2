import shutil
import random
import os

def train_val(traindir, valdir):
    names = os.listdir(traindir)
    n = len(names)
    valnum = n // 10
    for i in range(valnum):
        names = os.listdir(traindir)
        print(len(names))
        fac = random.choice(names)
        readfile = os.path.join(traindir, fac)
        writefile = os.path.join(valdir, fac)
        shutil.move(readfile, writefile)

if __name__ == '__main__':
    trainroot = '/defaultShare/share/wujl/83/ddb_classall/train'
    labels = ['17', '18']
    valroot = '/defaultShare/share/wujl/83/ddb_classall/val'
    for label in labels:
        print(label)
        traindir = os.path.join(trainroot, label)
        valdir = os.path.join(valroot, label)
        os.makedirs(valdir, exist_ok=True)
        train_val(traindir, valdir)
