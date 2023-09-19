import math
import h5py
import numpy as np
import pywt
import EMGImg
from Preprocess.EnhanceZsc import actionSeg


def emg_dwpt2d(signal, wavelet_name='db1'):
    wavelet_level = 3
    wp = pywt.WaveletPacket2D(signal, wavelet_name, mode='sym')
    coeffs = []
    level_coeff = wp.get_level(wavelet_level)
    for i in range(len(level_coeff)):
        coeffs.append(level_coeff[i].data.flatten())
    coeffs = np.hstack(coeffs)
#    coeffs = coeffs.flatten()
    return coeffs

def featureRMS(data):
    return np.sqrt(np.mean(data**2, axis=0))

def featureMAV(data):
    return np.mean(np.abs(data), axis=0) 

def featureWL(data):
    return np.sum(np.abs(np.diff(data, axis=0)),axis=0)/data.shape[0]

def featureZC(data, threshold=10e-7):
    numOfZC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(1,length):
            diff = data[j,i] - data[j-1,i]
            mult = data[j,i] * data[j-1,i]
            
            if np.abs(diff)>threshold and mult<0:
                count=count+1
        numOfZC.append(count/length)
    return np.array(numOfZC)

def featureSSC(data,threshold=10e-7):
    numOfSSC = []
    channel = data.shape[1]
    length  = data.shape[0]
    
    for i in range(channel):
        count = 0
        for j in range(2,length):
            diff1 = data[j,i]-data[j-1,i]
            diff2 = data[j-1,i]-data[j-2,i]
            sign  = diff1 * diff2
            
            if sign>0:
                if(np.abs(diff1)>threshold or np.abs(diff2)>threshold):
                    count=count+1
        numOfSSC.append(count/length)
    return np.array(numOfSSC)


def getfeature(seglist):
    Datalist=[]
    all_fea = []  # 存放每个动作的emg对象
    imageLength = 20
    stridewindow = 1

    for i in range(len(seglist)):
        # 考虑信号放大到伏特
        iemg = seglist[i].data
        length = int(math.floor((iemg.shape[0] - imageLength) / stridewindow) + 1)
        rms = [featureRMS(iemg[stridewindow * j:stridewindow * j + imageLength, :]) for j in range(length)]
        feature_data= np.array(rms)
        all_fea.extend(feature_data)
        all_fea= np.array(all_fea)

        #封装特征图
        fea_len = int(math.floor((all_fea.shape[0] - imageLength)) / stridewindow + 1)
        for j in range(fea_len):
            emg2 = all_fea[stridewindow * j:stridewindow * j + imageLength, :]  # 连续分割方式
            label2 = int(seglist[i].label[0]) - 1
            rep2 = (seglist[i].rep[0])
            myemg = EMGImg.reEMG(emg2, label2, rep2)
            Datalist.append(myemg)




    return all_fea


if __name__ == '__main__':
    for j in range(1, 2):
        h5 = h5py.File('F:/DB2/Downfilter/DB2_s' + str(j) + 'down.h5', 'r')
        alldata = h5['alldata'][:]
        # 动作状态数据分割  肌电子图标准化
        actionlist = actionSeg(alldata, 2, 12)

        all_fea = getfeature(actionlist)

        file = h5py.File('F:/DB2/feature/MAV/DB2_s' + str(j) + 'mav.h5', 'w')

        file.create_dataset('mav_fea', data=(all_fea).astype('float32'))
        # file.create_dataset('Data1', data=(datalist[1]).astype('float32'))
        # file.create_dataset('Data2', data=(datalist[2]).astype('float32'))
        # file.create_dataset('Data3', data=(datalist[3]).astype('float32'))
        # file.create_dataset('Data4', data=(datalist[4]).astype('float32'))
        # file.create_dataset('Data5', data=(datalist[5]).astype('float32'))
        # file.create_dataset('label0', data=(labellist[0]).astype('int32'))
        # file.create_dataset('label1', data=(labellist[1]).astype('int32'))
        # file.create_dataset('label2', data=(labellist[2]).astype('int32'))
        # file.create_dataset('label3', data=(labellist[3]).astype('int32'))
        # file.create_dataset('label4', data=(labellist[4]).astype('int32'))
        # file.create_dataset('label5', data=(labellist[5]).astype('int32'))
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')


