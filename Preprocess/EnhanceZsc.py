import math
from copy import deepcopy
import h5py
import numpy as np
from sklearn import preprocessing
import EMGImg
import tsaug
from pyts.image import MarkovTransitionField, GramianAngularField
from Util.feature_utils import featureRMS, featureMAV, featureWL, featureZC, featureSSC


def OneEnhanceEMG(data):
#     data =data.T    #需要把数据转换成（channel，time）
    newdata =tsaug.TimeWarp(n_speed_change=1, max_speed_ratio=1.5, seed=123).augment(data)
#     enhancedata =np.hstack((data, newdata))
    return data, newdata


'''
    Step1: 按动作次数将手势分割（数据，手势类别，标签通道数）
'''
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



'''Step2: 只做数据增强'''
def EnhanceSeg(seglist):
    channel = 12
    Endatalist = []
    for i in range(len(seglist)):
        data = []
        newdata = []
        rep =(seglist[i].rep)[0]
        if rep in ([1, 3, 4, 6]):   # 增强前后的手势分开标准化，对训练集第1，3，4，6次进行增强
            for k in range(channel):
                tempdata=seglist[i].data
                datasample, newdatasample = OneEnhanceEMG(tempdata[:, k])
                data.append(datasample)
                newdata.append(newdatasample)
            data = (np.array(data)).T
            newdata = (np.array(newdata)).T
            Endata = np.vstack((data, newdata))
            Enlabel = np.hstack((seglist[i].label, seglist[i].label))
            Enrep = np.hstack((seglist[i].rep, seglist[i].rep))

        else:
            Endata = seglist[i].data
            Enlabel= seglist[i].label
            Enrep = seglist[i].rep
        myemg = EMGImg.reEMG(Endata, Enlabel,Enrep)
        Endatalist.append(myemg)
    return Endatalist



'''
    Step3: 分割为肌电子图
'''
def creatimg(segList):
    Datalist = []  #存放每个动作的emg对象
    imageLength = 400
    stridewindow = 100
    for i in range(len(segList)):
        # 构造肌电子图并将信号放大到伏特级别
        BNdata = (segList[i].data)
        length = math.floor((BNdata.shape[0] - imageLength) / stridewindow) + 1
        for j in range(length):
            emg2 = BNdata[stridewindow * j:stridewindow * j + imageLength, :]
            label2 = int(segList[i].label[0]) - 1
            rep2 =(segList[i].rep[0])
            myemg = EMGImg.reEMG(emg2, label2, rep2)
            Datalist.append(myemg)

    return Datalist

#特征图拼接方法
def feaimg(segList):
    Datalist = []  #存放每个动作的emg对象
    imageLength = 20
    stridewindow = 1
    for i in range(len(segList)):
        # 考虑构造肌电子图并将信号放大到伏特级别
        iemg = (segList[i].data)*1e6
        length = math.floor((iemg.shape[0] - imageLength) / stridewindow) + 1
        for j in range(length):
            rms = featureRMS(iemg[stridewindow * j:stridewindow * j + imageLength, :])
            mav = featureMAV(iemg[stridewindow * j:stridewindow * j + imageLength, :])
            wl = featureWL(iemg[stridewindow * j:stridewindow * j + imageLength, :])
            zc = featureZC(iemg[stridewindow * j:stridewindow * j + imageLength, :])
            ssc = featureSSC(iemg[stridewindow * j:stridewindow * j + imageLength, :])
            featureStack = np.hstack((rms, mav, wl, zc, ssc))
            emg2 = featureStack  # 连续分割方式
            label2 = int(segList[i].label[0]) - 1
            rep2 = (segList[i].rep[0])
            myemg = EMGImg.reEMG(emg2, label2, rep2)
            Datalist.append(myemg)
    return Datalist



def GASF(segList):
    Datalist = []  # 存放每个动作的emg对象
    imageLength = 20
    stridewindow = 1
    for i in range(len(segList)):
        # 考虑构造肌电子图并将信号放大到伏特级别
        BNdata = (segList[i].data)
        length = math.floor((BNdata.shape[0] - imageLength) / stridewindow) + 1
        for j in range(length):
            emg = BNdata[stridewindow * j:stridewindow * j + imageLength, :]  # 连续分割方式
            gasf = GramianAngularField(image_size=10, method='summation')
            emg2 = gasf.fit_transform(emg)
            label2 = int(segList[i].label[0]) - 1
            rep2 = (segList[i].rep[0])
            myemg = EMGImg.reEMG(emg2, label2, rep2)
            Datalist.append(myemg)
    return Datalist


'''Step4: 数据归一化'''
def uniform(seglist,basis):
    channel = 12
    Datalist = []
    for i in range(len(seglist)):
        iemg = seglist[i].data
        emg2 = basis.transform(iemg)
        label2 = seglist[i].label
        rep2 = (seglist[i].rep)
        myemg = EMGImg.reEMG(emg2, label2, rep2)
        Datalist.append(myemg)
    return Datalist


'''Step5: 数据集划分'''
def getdata(seglist):
    emglist = [[], [], [], [], [], []]
    labellist = [[], [], [], [], [], []]
    for i in range(len(seglist)):
        for j in ([1, 2, 3, 4, 5, 6]):
            if (seglist[i].rep) == j:
                emglist[j - 1].append(seglist[i].data)
                labellist[j - 1].append(seglist[i].label)
    # 按重复次数分布存入对应数组
    for i in range(len(emglist)):
        emglist[i] = np.array(emglist[i])
    for i in range(len(labellist)):
        labellist[i] = np.array(labellist[i])

    return emglist, labellist







if __name__ == '__main__':
    for j in range(1, 2):
        h5 = h5py.File('F:/DB2/filter/DB2_s' + str(j) + 'filter.h5', 'r')
        alldata = h5['alldata'][:]
        #对所有数据归一化，得到均值
        basis = preprocessing.MinMaxScaler().fit(alldata[:, 0:12])
        #动作状态数据分割  肌电子图标准化
        actionlist = actionSeg(alldata, 49, 12)
        # Enhancelist = EnhanceSeg(actionlist)
        imglist = creatimg(actionlist)
        unilist = uniform(imglist, basis)
        datalist, labellist= getdata(unilist)


        file = h5py.File('F:/DB2/Part/DB2_s' + str(j) + 'Seg.h5', 'w')

        file.create_dataset('Data0', data=(datalist[0]).astype('float32'))
        file.create_dataset('Data1', data=(datalist[1]).astype('float32'))
        file.create_dataset('Data2', data=(datalist[2]).astype('float32'))
        file.create_dataset('Data3', data=(datalist[3]).astype('float32'))
        file.create_dataset('Data4', data=(datalist[4]).astype('float32'))
        file.create_dataset('Data5', data=(datalist[5]).astype('float32'))
        file.create_dataset('label0', data=(labellist[0]).astype('int32'))
        file.create_dataset('label1', data=(labellist[1]).astype('int32'))
        file.create_dataset('label2', data=(labellist[2]).astype('int32'))
        file.create_dataset('label3', data=(labellist[3]).astype('int32'))
        file.create_dataset('label4', data=(labellist[4]).astype('int32'))
        file.create_dataset('label5', data=(labellist[5]).astype('int32'))
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')
