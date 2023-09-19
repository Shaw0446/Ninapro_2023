import math
from copy import deepcopy
import h5py
import numpy as np
from sklearn import preprocessing

import EMGImg

# # 直接构造成图像数据，数据为list，元素为二维array
def bncreatimg(segList):
    emglist = [[], [], [], [], [], []]
    labellist = [[], [], [], [], [], []]
    imageLength = 400
    stridewindow = 100

    for i in range(len(segList)):
        # 在重复进行的手势中，指定第几次手势为测试集
        k = i % 6
        iemg = segList[i].data
        BNdata = []
        for m in range(12):
            # 均值
            average = float(sum(iemg[:, m])) / len(iemg[:, m])
            # 方差
            total = 0
            for value in iemg[:, m]:
                total += (value - average) ** 2
            stddev = math.sqrt(total / len(iemg[:, m]))
            # z-score标准化方法
            BNdata.append([(x - average) / stddev for x in iemg[:, m]])
        BNdata = (np.array(BNdata)).T
        length = math.floor((BNdata.shape[0] - imageLength) / stridewindow) + 1
        for j in range(length):
            subImage = BNdata[stridewindow * j:stridewindow * j + imageLength, :]  # 连续分割方式
            emglist[k].append(subImage)
            index = int(segList[i].label[0]) - 1
            labellist[k].append(index)
    for i in range(len(emglist)):
        emglist[i] = np.array(emglist[i])
    for i in range(len(labellist)):
        labellist[i] = np.array(labellist[i])

    return emglist, labellist


# 按动作次数将手势分割（数据，手势类别，通道数）
def actionSeg(data, dataclass, channel):
    data = np.array(data)
    Datalist = []
    for j in range(1, dataclass + 1):
        num = 0
        while (num < 6):
            for i in range(len(data)):
                if data[i, channel] == j:  # alldata最后一维存储标签
                    start = i
                    break
            for k in range(start, len(data)):
                if data[k, channel] == 0:
                    end = k - 1
                    break
                if k == len(data) - 1:
                    end = len(data) - 1
                    break

            print(start, end)
            num = num + 1
            emg = (deepcopy(data[start:end, 0:channel]))
            label = (deepcopy(data[start:end, channel]))
            rep = (deepcopy(data[start:end, channel + 1]))
            myemg = EMGImg.reEMG(emg, label, rep)
            Datalist.append(myemg)  # 用append将每次动作分割为单独的元素,保存了emg数据和标签
            for m in range(start, end + 1):
                data[m, channel] = 100
    return Datalist  # 返回数据为list类型，保存不同长度的数据


'''
    不分割为肌电子图返回数据标准化后结果,BNdatalist元素为单独一个动作手势标准化后的结果，
保存有对应的标签 ,返回类型(time,channel)
'''
def bnsegment(seglist):
    channel=12
    BNdatalist=[]
    for i in range(len(seglist)):
        iemg = seglist[i].data
        BNdata = preprocessing.StandardScaler().fit_transform(iemg)
        myemg = EMGImg.EMG(BNdata, seglist[i].label)
        BNdatalist.append(myemg)
    return BNdatalist


def creatimg(segList):
    emglist = [[], [], [], [], [], []]
    labellist = [[], [], [], [], [], []]
    imageLength = 400
    stridewindow = 100

    for i in range(len(segList)):
        # 在重复进行的手势中，指定第几次手势为测试集
        k = i % 6
        BNdata = segList[i].data
        length = math.floor((BNdata.shape[0] - imageLength) / stridewindow) + 1
        for j in range(length):
            subImage = BNdata[stridewindow * j:stridewindow * j + imageLength, :]  # 连续分割方式
            emglist[k].append(subImage)
            index = int(segList[i].label[0]) - 1
            labellist[k].append(index)
    for i in range(len(emglist)):
        emglist[i] = np.array(emglist[i])
    for i in range(len(labellist)):
        labellist[i] = np.array(labellist[i])

    return emglist, labellist


if __name__ == '__main__':
    for j in range(1, 41):
        h5 = h5py.File('F:/DB2/filter/DB2_s' + str(j) + 'filter.h5', 'r')
        alldata = h5['alldata']
        seglist = actionSeg(alldata,2, 12)
        bnlist = bnsegment(seglist)
        datalist, labellist = creatimg(bnlist)

        file = h5py.File('F:/DB2/SegZsc/DB2_s' + str(j) + 'allseg400100mZsc.h5', 'w')
        file.create_dataset('Data0', data=datalist[0])
        file.create_dataset('Data1', data=datalist[1])
        file.create_dataset('Data2', data=datalist[2])
        file.create_dataset('Data3', data=datalist[3])
        file.create_dataset('Data4', data=datalist[4])
        file.create_dataset('Data5', data=datalist[5])
        file.create_dataset('label0', data=labellist[0])
        file.create_dataset('label1', data=labellist[1])
        file.create_dataset('label2', data=labellist[2])
        file.create_dataset('label3', data=labellist[3])
        file.create_dataset('label4', data=labellist[4])
        file.create_dataset('label5', data=labellist[5])
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')
