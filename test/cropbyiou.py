from matting import parse_xml
from matting import matting
import os
import torch
import torchvision
from PIL import Image
import cv2
import xml.etree.ElementTree as ET

def parse_prexml(xmlpath, thresh):
    boxes = []
    tags = []
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    for obj in root.iter("object"):
        score = float(obj.find("score").text)
        if score <= thresh:
            continue
        if obj.find("name").text != 'ddb_others':
            continue
        box = []
        tags.append(obj.find("name").text)

        xmlbox = obj.find('bndbox')
        box.append(float(xmlbox.find("xmin").text))
        box.append(float(xmlbox.find("ymin").text))
        box.append(float(xmlbox.find("xmax").text))
        box.append(float(xmlbox.find("ymax").text))
        boxes.append(box)
    return boxes, tags


def calculatexml(srcpath):
    xmlpaths = os.listdir(srcpath)
    boxes = 0
    names = []
    for xmlpath in xmlpaths:
        path = os.path.join(srcpath, xmlpath)
        rects, tags = parse_xml(path)
        boxes += len(rects)
        names.extend(tags)
    print(boxes)
    print(names)
    return boxes, names


def txt2box(txtpath):
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        box = []
        line = line.strip()
        locals = line.split(' ')
        for local in locals:
            coor = local.split(':')[-1]
            if ']' in coor:
                box.append(float(coor[0:4]))
            else:
                box.append(float(coor))
        boxes.append(box)
    return boxes

def iou(box1, box2):
    area1 = (box1[2] - box1[0])*(box1[3] - box1[1])
    area2 = (box2[2] - box2[0])*(box2[3] - box2[1])
    topx = max(box1[0], box2[0])
    topy = max(box1[1], box2[1])
    downx = min(box1[2], box2[2])
    downy = min(box1[3], box2[3])
    if (downx - topx)>0 and (downy - topy) > 0:
        inner = (downx-topx)*(downy - topy)
    else:
        inner = 0
    return inner/(area1+area2-inner)

def usefulboxes(txtroot, xmlroot, thresh, score):
    txtnames = os.listdir(txtroot)
    usefuldic = {}
    for txtname in txtnames:
        txtpath = os.path.join(txtroot, txtname)
        xmlname = txtname[0:-4] + '.xml'
        xmlpath = os.path.join(xmlroot, xmlname)
        if not os.path.exists(xmlpath):
            continue
        xmlboxes, xmlnames = parse_xml(xmlpath)
        txtboxes, prexmlnames = parse_prexml(txtpath, score)
        if len(txtboxes) == 0:
            continue
   #     for txtbox in txtboxes:
        repeat = []
        for txtbox in txtboxes:
            for i in range(len(xmlboxes)):
                xmlbox = xmlboxes[i]
                if xmlbox in repeat:
                    continue
                if iou(txtbox, xmlbox) > thresh:
                    repeat.append(xmlbox)
                    a = txtbox.copy()
                    a.append(xmlnames[i])
                    print(txtbox)
                    if txtname not in usefuldic.keys():
                        usefuldic[txtname] = [a]
                    else:
                        usefuldic[txtname].append(a)
                    break
    return usefuldic


def aboxes(txtroot, score):
    txtnames = os.listdir(txtroot)
    dic = {}
    for txtname in txtnames:
        txtpath = os.path.join(txtroot, txtname)
        txtboxes, txtnames = parse_prexml(txtpath, score)
        if len(txtboxes) == 0:
            continue
        for txtbox in txtboxes:
            if txtname not in dic.keys():
                dic[txtname] = [txtbox]
            else:
                dic[txtname].append(txtbox)
    return dic


def cropboxbydic(dic, imgdir, savepath, txtpath, recall):
    for key in dic.keys():
        imgname = key[0:-4]+'.jpeg'
        imgpath = os.path.join(imgdir, imgname)
        if not os.path.exists(imgpath):

            imgname = key[0:-4] + '.jpg'
            imgpath = os.path.join(imgdir, imgname)
            if not os.path.exists(imgpath):
                continue
        img = cv2.imread(imgpath)
        infors = dic[key]
        for infor in infors:
            box = [[infor[0], infor[1], infor[2], infor[3]]]
            matimg = matting(box, img)

            matimg = matimg[0]
            savename = key[0:-4] + '_{}_{}_{}_{}.jpg'.format(infor[0], infor[1], infor[2], infor[3])
            savedir = os.path.join(savepath, savename)
            print(savedir)
            cv2.imwrite(savedir, matimg)
            if recall:
                with open(txtpath, 'a', encoding='utf-8') as f:
                    f.write('{}, {}'.format(savename, infor[-1])+'\n')
            else:
                with open(txtpath, 'a', encoding='utf-8') as f:
                    f.write('{}'.format(savename) +'\n')

# score is  the detection score
def getpre(project, score):
    path = '/defaultShare/share/wujl/83/test/{}/originpic/det_xml/'.format(project)
    imgdir = '/defaultShare/share/wujl/83/test/{}/originpic/JPEGImages'.format(project)
    savepath = '/defaultShare/share/wujl/83/test/{}/croppic/croppre_{}'.format(project, score)
    os.makedirs(savepath, exist_ok=True)
    txtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/{}_pre.txt'.format(project, score)
    #txtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/justtestpre.txt'.format(project)
    d = aboxes(path, score)
    cropboxbydic(d, imgdir, savepath, txtpath, recall=False)

# thresh is the iou, score is the detection score
def getrecall(project, score, thresh):
    path = '/defaultShare/share/wujl/83/test/{}/originpic/det_xml/'.format(project)
    xmlpath = '/defaultShare/share/wujl/83/test/{}/originpic/Annotations/'.format(project)
    imgdir = '/defaultShare/share/wujl/83/test/{}/originpic/JPEGImages'.format(project)
    txtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/recall_{}_{}.txt'.format(project, score, thresh)
    savepath = '/defaultShare/share/wujl/83/test/{}/croppic/croprecall_{}_{}'.format(project, score, thresh)
    os.makedirs(savepath, exist_ok=True)
    d = usefulboxes(path, xmlpath, thresh, score)
    cropboxbydic(d, imgdir, savepath, txtpath, recall=True)

if __name__ == '__main__':
    project = '9_21'
    thresh = 0.3
    scores = [0.6]
    for score in scores:
        getpre(project, score)
        print(score)
    getrecall(project, score, thresh)