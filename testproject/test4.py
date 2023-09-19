import h5py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


# 滤波信号保留10-500hz
def Butterfilteremg(sig):
    b, a = signal.butter(4, [0.01, 0.5], 'bandpass')
    y = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return y


'''
设计并绘制滤波器以从以 200 Hz 采样的信号中去除 60 Hz 分量，使用品质因数 Q = 30
'''


def Trapwave(sig):
    fs = 500.0  # Sample frequency (Hz)
    f0 = 50.0  # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    # Design notch filter
    b, a = sig.iirnotch(f0, Q, fs)
    # Frequency response
    freq, h = sig.freqz(b, a, fs=fs)
    return freq


if __name__ == '__main__':
    for j in range(1, 3):
        file = h5py.File('F:/DB2/EnhanceSegZsc/DB2_s' + str(j) + 'allSegZsc.h5', 'r')
        # 将六次重复手势分开存储
        Data0, Data1, Data2, Data3, Data4, Data5 = file['Data0'][:], file['Data1'][:], file['Data2'][:] ,file['Data3'][:], file['Data4'][:],  file['Data5'][:]
        Label0, Label1, Label2, Label3, Label4, Label5 = file['label0'][:], file['label1'][:], file['label2'][:] , file['label3'][:], file['label4'][:], file['label5'][:]
        file.close()



    # # 存储为h5文件
    #     file = h5py.File('F:/DB2/refilter/DB2_s' + str(j) + 'refilter.h5', 'w')
    #     # file.create_dataset('emgdata', data=filteremg)
    #     # file.create_dataset('label', data=label)
    #     file.create_dataset('alldata', data=alldata)
    #     file.close()