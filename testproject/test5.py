from copy import deepcopy
import import_ipynb
import numpy as np
import tsaug
import h5py
import EMGImg
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化
# 按动作次数将手势分割（数据，手势类别，通道数）


def segment(data, dataclass, channel):
    data = np.array(data)
    Datalist = []
    for j in range(1, dataclass + 1):
        num = 0
        while (num < 6):
            for i in range(data.shape[0]):
                if data[i, channel] == j:  # alldata最后一维存储标签
                    start = i
                    break
            for k in range(start, data.shape[0]):
                if data[k, channel] == 0:
                    end = k
                    break
                if k == len(data) - 1:
                    end = data.shape[0]
                    break

            print(start, end)
            num = num + 1
            emg = (deepcopy(data[start:end, 0:channel]))
            label = (deepcopy(data[start:end, channel]))
            myemg = EMGImg.EMG(emg, label)
            Datalist.append(myemg)  # 用append将每次动作分割为单独的元素,保存了emg数据和标签
            for m in range(start, end):
                data[m, channel] = 100
    return Datalist  # 返回数据为list类型，保存不同长度的数据


'''该函数在输入二维数组时会将各个通道随机打乱，破坏了不同电极采样的同步性，故设定随机种子，单个通道依次打乱'''
def OneenhanceEMG(data):
    newdata =tsaug.TimeWarp(n_speed_change=1, max_speed_ratio=2, seed=123).augment(data)
#     enhancedata =np.hstack((data, newdata))
    return data, newdata

for j in range(1, 2):
    h5 = h5py.File('F:/DB2/filter/DB2_s' + str(j) + 'filter.h5', 'r')
    alldata = h5['alldata']
    seglist = segment(alldata, 2, 12)
    timedata=seglist[0].data
    data=[]
    newdata=[]
    for k in range(12):
            datasample, newdatasample =OneenhanceEMG(timedata[:,k])
            data.append(datasample)
            newdata.append(newdatasample)
    data=np.array(data)
    newdata=np.array(data)
    plt.figure(figsize=(20,8))
    for i in range(12):
        plt.subplot(12,1,i+1)
        plt.plot(data[:,i],color=mcolors.TABLEAU_COLORS[colors[int(math.fabs(i-2))]])
    #         plt.xticks([])  #去掉x轴
        plt.axis('off')
    plt.show()

# for j in range(1, 2):
#     h5 = h5py.File('F:/DB2/filter/DB2_s' + str(j) + 'filter.h5', 'r')
#     alldata = h5['alldata']
#     seglist = segment(alldata,2, 12)
#     timedata=seglist[0].data
#
#     data, newdata =enhanceEMG((timedata))
#
#
#     plt.figure(figsize=(20,8))
#     for i in range(12):
#         plt.subplot(12,1,i+1)
#         plt.plot(data[:,i],color=mcolors.TABLEAU_COLORS[colors[int(math.fabs(i-2))]])
#         #         plt.xticks([])  #去掉x轴
#         plt.axis('off')
#     plt.show()
