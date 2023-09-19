import h5py
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nina_funcs as nf

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))
image_size = 16
channal = 12

# Two-dimensional discrete features(using Cartesian Product)
def two_dimension_graph(feature):
    feature = feature.reshape((feature.shape[0], -1, channal))    #转换为三维
    feature = feature.transpose(0, 2, 1)        #交换位置后变为（samples, channel, imagesize）
    feature_graph = np.zeros((feature.shape[0], image_size, image_size, channal), dtype='float32')
    for i in range(feature.shape[0]):
        for j in range(feature.shape[1]):
            # Cartesian Product
            single_use = feature[i, j, :].reshape(-1, image_size)
            single_graph = (single_use. T * single_use)
            feature_graph[i, :, :, j] = single_graph
    # feature_graph = feature_graph.reshape((feature.shape[0], -1))
    return feature_graph

'''设置信号投影方式'''
def sigmoid(single_feature):
    output_s = 0.5(single_feature)
    return output_s


for j in range(1, 2):
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/down_Fea/DB2_s' + str(j) + 'fea.h5', 'r')
    '''step1: 获取特征数组'''
    x_train, y_train = file['x_train'][:], file['y_train'][:]
    x_test, y_test = file['x_test'][:], file['y_test'][:]
    file.close()

    '''step2: 特征图像转换'''
    #选择特征组合
    X_train_high = (x_train.T[::2]).T
    X_train_low = (x_train.T[1::2]).T
    X_test_high = (x_test.T[::2]).T
    X_test_low = (x_test.T[1::2]).T

    X_train_highimg = two_dimension_graph(X_train_high)
    X_train_lowimg = two_dimension_graph(X_train_low)
    X_test_highimg = two_dimension_graph(X_test_high)
    X_test_lowimg = two_dimension_graph(X_test_low)




    # 存储为h5文件
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/featureMap/DB2_s' + str(j) + 'map.h5', 'w')
    file.create_dataset('X_train_highimg', data=X_train_highimg.astype('float32'))
    file.create_dataset('X_train_lowimg', data=X_train_lowimg.astype('float32'))
    file.create_dataset('X_test_highimg', data=X_test_highimg.astype('float32'))
    file.create_dataset('X_test_lowimg', data=X_test_lowimg.astype('float32'))

    file.create_dataset('y_train', data=y_train.astype('int32'))
    file.create_dataset('y_test', data=y_test.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '特征图构建完成***********************')

