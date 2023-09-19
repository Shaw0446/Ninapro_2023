from copy import deepcopy
import h5py
import numpy as np


#数据集分为十二类
def Sep12Data(data):
    data1, data2, data3, data4, data5, data6, data7, data8, data9, data10 \
        , data11, data12 = [], [], [], [], [], [], [], [], [], [], [], [],

    for i in range(len(data)):
        data1.append(deepcopy(data[i, :, 0]))
        data2.append(deepcopy(data[i, :, 1]))
        data3.append(deepcopy(data[i, :, 2]))
        data4.append(deepcopy(data[i, :, 3]))
        data5.append(deepcopy(data[i, :, 4]))
        data6.append(deepcopy(data[i, :, 5]))
        data7.append(deepcopy(data[i, :, 6]))
        data8.append(deepcopy(data[i, :, 7]))
        data9.append(deepcopy(data[i, :, 8]))
        data10.append(deepcopy(data[i, :, 9]))
        data11.append(deepcopy(data[i, :, 10]))
        data12.append(deepcopy(data[i, :, 11]))

    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)
    data4 = np.array(data4)
    data5 = np.array(data5)
    data6 = np.array(data6)
    data7 = np.array(data7)
    data8 = np.array(data8)
    data9 = np.array(data9)
    data10 = np.array(data10)
    data11 = np.array(data11)
    data12 = np.array(data12)

    # 重塑成（sample，1）
    data1 = data1.reshape(-1, 400, 1)
    data2 = data2.reshape(-1, 400, 1)
    data3 = data3.reshape(-1, 400, 1)
    data4 = data4.reshape(-1, 400, 1)
    data5 = data5.reshape(-1, 400, 1)
    data6 = data6.reshape(-1, 400, 1)
    data7 = data7.reshape(-1, 400, 1)
    data8 = data8.reshape(-1, 400, 1)
    data9 = data9.reshape(-1, 400, 1)
    data10 = data10.reshape(-1, 400, 1)
    data11 = data11.reshape(-1, 400, 1)
    data12 = data12.reshape(-1, 400, 1)

    return data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12


#数据集分为二类
def Sep2Data(data):
    data1, data2 = [], []
    for i in range(len(data)):
        data1.append((data[i, :, 0:8]))
        data2.append((data[i, :, 8:12]))
    data1 = np.array(data)
    data2 = np.array(data2)
    return data1, data2

#数据集分为三类
def Sep3Data(data):
    data1, data2, data3 = [], [], []
    for i in range(len(data)):
        data1.append((data[i, :, 0:8]))
        data2.append((data[i, :, 8:10]))
        data3.append((data[i, :, 10:12]))
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    return data1, data2, data3


def Sep3Fea(data):
    data1, data2, data3 = [], [], []
    for i in range(len(data)):
        data1.append((data[i, :, :, 0:8]))
        data2.append((data[i, :, :, 8:10]))
        data3.append((data[i, :, :, 10:12]))
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3 = np.array(data3)

    return data1, data2, data3



# if __name__ == '__main__':
#     file = h5py.File('../data/DB2_S1segment2e4.h5', 'r')
#     X_train = file['trainData'][:]
#     X_test = file['testData'][:]
#
#     Xtrain1, Xtrain2, Xtrain3=Sep3trainData(X_train)
