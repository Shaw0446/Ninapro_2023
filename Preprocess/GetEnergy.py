import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np
from mpl_toolkits.axisartist import SubplotZero
from tqdm import tqdm


def rect_sum(amplitude, v):
    '''绘制矩形'''
    xmax = np.max(amplitude)
    xmin = np.min(amplitude)
    xavg = np.average(amplitude)
    ymax = np.max(v)
    ymin = np.min(v)
    yavg = np.average(v)

    N1 = 20
    N2 = 20

    w = (xmax - xmin) / N1
    l = (ymax - ymin) / N2

    '''target存储了相图坐标, GSMatrix为灰度矩阵'''
    GSMatrix = np.zeros([N1, N2], dtype=np.float32)
    targets = list(zip(amplitude, v))
    for target in targets:
        X_Matrix = int((target[0]-xmin)/w)-1
        Y_Matrix = int((target[1]-ymin)/l)-1
        GSMatrix[X_Matrix, Y_Matrix] = GSMatrix[X_Matrix, Y_Matrix] + 1

    '''2D移动均值法'''
    GSMatrix_Mov = np.zeros([N1, N2], dtype=np.float32)
    for i in range(1,N1-1):
        for j in range(1, N2 - 1):
            GSMatrix_Mov[i, j] = (GSMatrix[i - 1, j - 1] + GSMatrix[i - 1, j] + GSMatrix[i - 1, j + 1]
                                  + GSMatrix[i, j - 1] + GSMatrix[i, j] + GSMatrix[i, j + 1]
                                  + GSMatrix[i + 1, j - 1] + GSMatrix[i + 1, j] + GSMatrix[i + 1, j + 1]) / 9


    '''I表示为加入权重的灰度矩阵，w * l为不同相图的单元格面积'''
    I = GSMatrix_Mov * (w * l)
    # B = I.astype(np.int)

    return I


def channel_join(emg_arr, channel=12):
    N1 = 20
    N2 = 20
    splice_matrix = []
    for iemg in (tqdm(emg_arr)):
        # 多通道灰度图的拼接占位
        join_map = np.zeros([N1 * 3, N2 * 4], dtype=np.float32)
        one_matrix = []
        for i in range(channel):
            amplitude, v = energy(iemg[:, i])
            one_map = rect_sum(amplitude, np.array(v))
            #生成60*80的拼接矩阵
            a = int(i / 4)
            b = i % 4
            join_map[a * N1:(a + 1) * N1, b * N2:(b + 1) * N2] = one_map

            one_matrix.append(rect_sum(amplitude, np.array(v)))
        splice_matrix.append(join_map)
    aaaa = np.array(splice_matrix)
    return np.array(splice_matrix)


def energy(iemg):
    v = []
    for i in range(len(iemg)):
        if i == 0:
            sum = iemg[i] - 0
        else:
            sum = iemg[i] - iemg[i - 1]
        v.append(sum.astype(np.float32))
    return iemg, v



def energy_map(iemg, freq=2000):
    y = []
    for i in range(len(iemg)):
        if i == 0:
            sum = iemg[i] - 0
        else:
            sum = iemg[i] - iemg[i - 1]
        y.append(sum)
    y2 = np.array(y)

    fig = plt.figure(1, (10, 6))
    ax = SubplotZero(fig, 1, 1, 1)
    fig.add_subplot(ax)

    """新建坐标轴"""
    ax.axis["xzero"].set_visible(True)
    # ax.axis["xzero"].label.set_text("y=0")
    # ax.axis["xzero"].label.set_color('green')
    ax.axis['yzero'].set_visible(True)
    # ax.axis["yzero"].label.set_text("x=0")

    """坐标箭头"""
    ax.axis["xzero"].set_axisline_style("-|>")
    ax.axis["yzero"].set_axisline_style("-|>")

    """隐藏坐标轴"""
    ax.axis["top", 'right', 'left', 'bottom'].set_visible(False)

    """设置刻度"""
    # ax.set_ylim(-3, 3)
    # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    # ax.set_xlim([-5, 8])
    # ax.set_xticks([-5,5,1])

    # 设置网格样式
    ax.grid(True, linestyle='-.')

    rect_sum(iemg[:], y)
    ax.scatter(iemg[:], y, s=4)
    # plt.plot(iemg[:], y, linestyle='-.')
    plt.show()


for j in range(1, 2):
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/df_Seg/DB2_s' + str(j) + 'Seg17.h5', 'r')
    x_train1, x_train3, x_train4, x_train6 = file['x_train1'][:],file['x_train3'][:],file['x_train4'][:],file['x_train6'][:]
    y_train1,y_train3,y_train4,y_train6 = file['y_train1'][:],file['y_train3'][:],file['y_train4'][:],file['y_train6'][:]
    x_test = file['x_test'][:]
    y_test = file['y_test'][:]
    file.close()

    # sample = x_train1[:50, :, :]
    map1 = channel_join(x_train1, 12)
    map3 = channel_join(x_train3, 12)
    map4 = channel_join(x_train4, 12)
    map6 = channel_join(x_train6, 12)
    maptest = channel_join(x_test,12)

    # 存储为h5文件
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/energy_map/DB2_s' + str(j) + 'map17.h5', 'w')
    file.create_dataset('map1', data=map1.astype('float32'))
    file.create_dataset('map3', data=map3.astype('float32'))
    file.create_dataset('map4', data=map4.astype('float32'))
    file.create_dataset('map6', data=map6.astype('float32'))
    file.create_dataset('y_train1', data=y_train1.astype('int'))
    file.create_dataset('y_train3', data=y_train3.astype('int'))
    file.create_dataset('y_train4', data=y_train4.astype('int'))
    file.create_dataset('y_train6', data=y_train6.astype('int'))

    file.create_dataset('maptest', data=maptest.astype('float32'))
    file.create_dataset('y_test', data=y_test.astype('int'))
    file.close()

    print('******************DB2_s' + str(j) + '完成***********************')

