#-*-coding:utf-8-*-

import matplotlib.pyplot as plt
import pywt
import math
import numpy as np
import h5py

h5 = h5py.File('../data/DB2_S1_image200.h5', 'r')
emg = h5['imageData']
imageLabel= h5['imageLabel']
data = []
coffs = []

for i in range(len(emg)):
    for j in range(200):
        Y = emg[i, j, 1]
        data.append(Y)
#create wavelet object and define parameters
w = pywt.Wavelet('db8') #选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data),w.dec_len)
print("maximum level is"+str(maxlev))
threshold= 0.04  #Threshold for filtering

#Decompose into wavelet components,to the level selected:
coffs = pywt.wavedec(data,'db8',level=maxlev) #将信号进行小波分解

for i in range(1,len(coffs)):
    temp = pywt.threshold(coffs[i],threshold*max(coffs[i]))
    coffs[i] = temp
datarec = pywt.waverec(coffs,'db8')#将信号进行小波重构

mintime = 0
maxtime = mintime+len(data)
windowstep = 200
print(mintime,maxtime)
for i in range(int(maxtime/windowstep)):
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(data[i*windowstep:(i+1)*windowstep])
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("Raw signal")
    plt.subplot(3, 1, 2)
    plt.plot(datarec[i*windowstep:(i+1)*windowstep])
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("De-noised signal using wavelet techniques")
    plt.subplot(3, 1, 3)
    plt.plot(data[i*windowstep:(i+1)*windowstep]-datarec[i*windowstep:(i+1)*windowstep])
    plt.xlabel('time (s)')
    plt.ylabel('error (uV)')
    plt.tight_layout()
    plt.show()
print(maxtime/windowstep)

