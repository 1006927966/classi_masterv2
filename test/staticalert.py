def getpicname(cropname):
    ip = cropname.split('_')[0]
    day = cropname.split('_')[1]
    return ip + '_' +day + '.jpg'



def getalertcount(txtpath, detthresh, classthresh, labels):
    repeat = []
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    total = len(lines)
    count = 0
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = factors[0]
        detscore = float(name.split('_')[-1][:-4])
        classscore = float(factors[-2])
        label = int(factors[-1])
        if label in labels:
            if detscore > detthresh:
                if classscore > classthresh:
                    picname = getpicname(name)
                    if picname not in repeat:
                        repeat.append(picname)
                    count += 1
    return count, count/total, len(repeat), len(repeat)/11666


if __name__ == '__main__':
    txtpath = '/defaultShare/share/wujl/83/online_data/2020-07-30_20_19.txt'
    labelboxes = [[2,3,5,6,7,10,14]]
    detthreshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    classthreshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for labels in labelboxes:
        for detthresh in detthreshs:
            for classthresh in classthreshs:
                count, rate, piccount, picrate = getalertcount(txtpath, detthresh, classthresh, labels)
                print('{}'.format(labels))
                print('det{}_c{} is : {}, {}, {}, {}'.format(detthresh, classthresh, count, rate, piccount, picrate))
                print('*'*10)

