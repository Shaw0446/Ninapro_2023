import math
import os
from copy import deepcopy
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

import EMGImg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
            Datalist.append(myemg)  # 用append将每次动作分割为单独的元素
            for m in range(start, end):
                data[m, channel] = 100
    return Datalist  # 返回数据为list类型，保存不同长度的数据


# # 构造成图像数据，数据为list，元素为二维array
def creatimg(segList):
    testData = []
    testLabel = []
    trainData = []
    trainLabel = []
    valiData = []
    valiLabel = []
    imageLength = 400
    stridewindow = 100

    for i in range(len(segList)):
        # 在重复进行的手势中，指定第几次手势为测试集
        k = i % 6
        # 第二次做测试集
        if k == 1:
            # iemg为单独一次的动作， imagelength采样窗口，将多通道emg数据连续分割
            # emg数据放大后方便训练识别
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
                testData.append(subImage)
                index = int(segList[i].label[0]) - 1
                testLabel.append(index)
        # 第五次做验证集
        if k == 4:
            # iemg为单独一次的动作， imagelength采样窗口，将多通道emg数据连续分割
            # emg数据放大后方便训练识别
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
                valiData.append(subImage)
                index = int(segList[i].label[0]) - 1
                valiLabel.append(index)
        # 其他做训练集
        else:
            # iemg为单独一次的动作
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
            # 拼接后转置为适应的数据形式
            BNdata = (np.array(BNdata)).T
            length = math.floor((BNdata.shape[0] - imageLength) / stridewindow) + 1
            for j in range(length):
                subImage = BNdata[stridewindow * j:stridewindow * j + imageLength, :]  # 连续分割方式
                trainData.append(subImage)
                index = int(segList[i].label[0]) - 1
                trainLabel.append(index)
    trainData = np.array(trainData)
    testData = np.array(testData)
    trainLabel = np.array(trainLabel)
    testLabel = np.array(testLabel)
    valiData = np.array(valiData)
    valiLabel = np.array(valiLabel)

    return trainData, trainLabel, testData, testLabel, valiData, valiLabel


if __name__ == '__main__':
    for j in range(15, 41):
        h5 = h5py.File('D:/Pengxiangdong/ZX/data/Originaldata/refilter/DB2_s' + str(j) + 'filter.h5', 'r')
        alldata = h5['alldata']
        segList = segment(alldata, 49, 12)
        trainData, trainLabel, testData, testLabel, valiData, valiLabel = creatimg(segList)

        file = h5py.File('D:/Pengxiangdong/ZX/data/reSegmentdata/reseg400100mValZsc/DB2_s' + str(j) + 'ValZsc.h5', 'w')
        file.create_dataset('trainData', data=trainData)
        file.create_dataset('trainLabel', data=trainLabel)
        file.create_dataset('testData', data=testData)
        file.create_dataset('testLabel', data=testLabel)
        file.create_dataset('valiData', data=valiData)
        file.create_dataset('valiLabel', data=valiLabel)
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')

# emg = segList[1].data
# iSampleRate=2000
# for i in range(12):
#     plt.subplot(12, 1, i + 1)
#     iSampleCount = len(emg)
#     t=np.linspace(0, iSampleCount / iSampleRate, iSampleCount)
#     plt.plot(t,emg[:, i])
#     # plt.ylim(-0.00025, 0.00025)
#     plt.xlabel("Time(s)")
#     # plt.xticks([])  # 去掉横坐标值
#     plt.yticks([])  # 去掉纵坐标值
#
# plt.show()
