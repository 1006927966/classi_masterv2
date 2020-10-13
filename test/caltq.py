#encoding=utf-8
import numpy as np
import os
import shutil
import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


def combinedic(onedic):
    cdic = {}
    standards = ['物料乱放', '物品乱放', "物料占道", "物料摆放不整齐", "垃圾", "地面脏", "others"]
    for standard in standards:
        cdic[standard] = 0
    total = 0
    for key in onedic.keys():
        total += onedic[key]
        if '物料乱放' in key:
            cdic['物料乱放'] += onedic[key]
        if '物品乱放' in key:
            cdic['物品乱放'] += onedic[key]
        if '物料混放' in key:
            cdic['物料乱放'] += onedic[key]
        if '物料占道' in key:
            cdic['物料占道'] += onedic[key]
        if '物料摆放不整齐' in key:
            cdic['物料摆放不整齐'] += onedic[key]
        if '垃圾' in key:
            cdic['垃圾'] += onedic[key]
        if '地面脏' in key:
            cdic['地面脏'] += onedic[key]
    cdic['others'] = total -  cdic['物料乱放'] - cdic['物品乱放'] - cdic['物料占道'] - cdic['物料摆放不整齐'] - cdic['垃圾'] -  cdic['地面脏']
    print(total)
    return cdic, total




# odic is origin, tdic is prediction
def calrecall(txtpath, prepath, maplist, savedir, imgdir, thresh, needlist, combine=True):
    odic = {}
    pdic = {}
    with open(txtpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = factors[-1]
        picname = factors[0]
        odic[picname] = name
    with open(prepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = maplist[int(factors[-1])]
        score = float(factors[1])
        savepath = os.path.join(savedir, name, factors[0])
        os.makedirs(os.path.join(savedir, name), exist_ok=True)
        readpath = os.path.join(imgdir, factors[0])
 #       shutil.cpdata(readpath, savepath)
        pdic[factors[0]] = [name, score]
# calculate overlab
    countdic = {}
    for key in pdic.keys():
        value = pdic[key]
        if (value[1] > thresh) and (value[0] in needlist):
            name = odic[key]
            if name not in countdic:
                countdic[name] = 1
            else:
                countdic[name] += 1
    if combine:
        cdic, total = combinedic(countdic)
        for key in cdic.keys():
            print('[*]! {} count is : {}'.format(key, cdic[key]))
        return total
    else:
        for key in countdic.keys():
            print('[*]! {} count is : {}'.format(key, countdic[key]))
        return 0



def calalert(pretxtpath, maplist, savedir, imgdir, thresh):
    dic = {}
    with open(pretxtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = maplist[int(factors[-1])]
        score = float(factors[1])
        if score <= thresh:
            continue
        if name not in dic:
            dic[name] = 1
        else:
            dic[name] += 1
        savepath = os.path.join(savedir, name, factors[0])
        os.makedirs(os.path.join(savedir, name), exist_ok=True)
        readpath = os.path.join(imgdir, factors[0])
        #shutil.cpdata(readpath, savepath)
    for key in dic.keys():
        print('[*]! {} count is : {}'.format(key, dic[key]))


def recallnum(needlist, iou, epoch, classiscore, maplist, project, detscore):
    croprecalltxt = '/defaultShare/share/wujl/83/test/{}/txtpath/recall_{}_{}.txt'.format(project, detscore, iou)
    prerecalltxt =  '/defaultShare/share/wujl/83/test/{}/txtpath/croprecall_{}_{}_{}.txt'.format(project, detscore, iou, epoch)
    croprecaldir = '/defaultShare/share/wujl/83/test/{}/croppic/croprecall_{}_{}'.format(project, detscore, iou)
    prerecallsavedir = '/defaultShare/share/wujl/83/test/{}/classisave/croprecall_{}_{}_{}'.format(project, detscore, iou, epoch)
    total = calrecall(croprecalltxt, prerecalltxt, maplist, prerecallsavedir, croprecaldir, classiscore, needlist)
    return total

def alertnum(maplist, classiscore, project, epoch, detscore):
    pretxtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/croppre_{}_{}.txt'.format(project, detscore, epoch)
    imgdir = '/defaultShare/share/wujl/83/test/{}/croppic/croppre_{}'.format(project, detscore)
    croppresavedir = '/defaultShare/share/wujl/83/test/{}/classisave/pre_{}_{}'.format(project, detscore, epoch)
    calalert(pretxtpath, maplist, croppresavedir, imgdir, classiscore)

if __name__ == '__main__':
    maplist = ['badcase', 'feigang', 'hanzha', 'laji00', 'laji01', 'laji02', 'laji03', 'luanfang', 'zangwu',
           'zhufei', 'xianshu_luan', 'xianshu_zhengqi', 'xiaojian_luan', 'xiaojian_zhengqi', 'pip_luan', 'pip_zhengqi', 'zhixiang',
               'bancai_luan', 'bancai_zhenqgi']
    ious = [0.3]
    needlist = ['xianshu_luan', 'luanfang',  'laji00', 'hanzha', 'laji03', 'laji02', 'pip_luan', 'bancai_luan']

    epochs = [20]
    project = '9_21'
    classiscores = [0]
    detscore = 0.6
    maxdic = {}
    for iou in ious:
        maxtotal = 0
        maxepoch = 0
        for epoch in epochs:
            for classiscore in classiscores:
                print('{}_{}_{}!!!!!!!!!'.format(epoch, iou, classiscore))
                total = recallnum(needlist, iou, epoch, classiscore, maplist, project, detscore)
                if total>maxtotal:
                    maxtotal = total
                    maxepoch = epoch
        maxdic[iou] = [maxepoch, maxtotal]
        print('maxtotal is {}!!!!!!!!!'.format(maxtotal))
        print('maxepoch is {}!!!!!!!!!'.format(maxepoch))
    print('*'*50)
    for key in maxdic.keys():
        maxepoch = maxdic[key][0]
        for classiscore in classiscores:
            print('{}_{}!!!!!!!!!'.format(maxepoch, classiscore))
            alertnum(maplist, classiscore, project, maxepoch, detscore)






