import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
#from lxml import etree, objectify

def parse_xml(xmlpath):
    boxes = []
    tags = []
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    for obj in root.iter("object"):
        box = []
        tags.append(obj.find("name").text)
        xmlbox = obj.find('bndbox')
        box.append(float(xmlbox.find("xmin").text))
        box.append(float(xmlbox.find("ymin").text))
        box.append(float(xmlbox.find("xmax").text))
        box.append(float(xmlbox.find("ymax").text))
        boxes.append(box)
    return boxes, tags

def matting(boxes, img):
    matting_imgs = []
    h, w = img.shape[:2]
    for box in boxes:
        y_pd = int((box[-1] - box[1])*0.1) + 1
        x_pd = int((box[-2] - box[0])*0.1) + 1
        y_min = int(box[1]) - y_pd
        y_max = int(box[-1]) + y_pd
        x_min = int(box[0]) - x_pd
        x_max = int(box[-2]) + x_pd

        if y_min <0 :
            y_min = 0
        if y_max > h:
            y_max = h
        if x_min < 0:
            x_min = 0
        if x_max>w:
            x_max = w
        matting_img = img[y_min:y_max, x_min:x_max]
        matting_imgs.append(matting_img)
    return matting_imgs


def matting_imgs(imgspath, xmlspath, savepath, day, numconstraint=3500, name_bool=False):
    imgnames = os.listdir(imgspath)
    count = 0
    for imgname in imgnames:
        if name_bool:
            if imgname not in name_bool:
                continue
        if count >= numconstraint:
            return 0
        if ('jpeg' not in imgname) and ('jpg' not in imgname):
            continue
        if 'jpeg' in imgname:
            id = imgname[:-5]
        if 'jpg' in imgname:
            id = imgname[:-4]
        imgpath = os.path.join(imgspath, imgname)
        xmlpath = os.path.join(xmlspath, id+'.xml')
        print(xmlpath)
        if not os.path.exists(xmlpath):
            print('xml is not exist!!!!')
            continue
        img = cv2.imread(imgpath)
        boxes, tags = parse_xml(xmlpath)
        boxnum = len(boxes)
        if boxnum == 0:
            continue
        for i in range(boxnum):
            count += 1
            box = boxes[i]
            savefilename = '{}_{}_{}_{}_{}.jpg'.format(id, box[0], box[1], box[2], box[3])
            savefilefolder = os.path.join(savepath, day)
            os.makedirs(savefilefolder, exist_ok=True)
            savefilepath = os.path.join(savefilefolder, savefilename)
            matting_imgs = matting([box], img)
            cv2.imwrite(savefilepath, matting_imgs[0])
        print('[*]! begin next pic')
    return 1





if __name__ == "__main__":
    name_bool = os.listdir("/defaultShare/share/wujl/83/online_data/labeldata/classi_labelv1")
    print(len(name_bool))
    days = ['2020-09-13']
    for day in days:
        imgspath = '/defaultShare/share/wujl/83/online_data/daydata/{}/JPEGImages'.format(day)
        xmlspath = '/defaultShare/share/wujl/83/online_data/daydata/{}/Annotations'.format(day)
        savepath = '/defaultShare/share/wujl/83/online_data/cropdata'
        p = matting_imgs(imgspath, xmlspath, savepath, day, numconstraint=20000000, name_bool=False)
        print('end')


