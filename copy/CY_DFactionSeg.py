import math
import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.signal as signal
import nina_funcs as nf
from tqdm import tqdm


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))

'''
    本类将上一个实验的预处理部分用dataframe格式重写,便于后续实验在
    数据处理上共用,也是更符合github工具包的存储格式
'''

'''  df形式的动作分割（要求能不排除休息状态）'''
def action_seg(data, channel, endlabel):
    actionList = []  # 存储动作
    tag = []  # 记录标签
    begin = 0
    for aim in tqdm(range(len(data) - 1)):
        # 控制手势停止时间
        if (data.iloc[aim, channel] ==endlabel+1):
            break;
        if (data.iloc[aim, channel] != data.iloc[aim + 1, 12]):
            tag.append(aim)
            end = aim
            actionList.append(data[begin:end])
            begin = end + 1
    actionList.append(data[begin:len(data)])
    return actionList

'''数据标准化（list的元素为df的方式）'''
def uniform(actionList, channel):
    BNdatalist = []
    for action in actionList:
        iemg = action.iloc[:, :channel].copy()
        labels = action.iloc[:, channel:].copy().reset_index()
        scaler = preprocessing.StandardScaler()
        BNemg = pd.DataFrame(scaler.fit_transform(iemg))
        temp = pd.concat([BNemg, labels], axis=1)
        BNdatalist.append(temp)

    return BNdatalist



'''时间增量窗口分割和划分数据集，输入参数为存储动作的list'''
def action_comb(actionList, timeWindow, strideWindow, channel=12):
    emgList = [[], [], [], [], [], [], []]
    labelList = [[], [], [], [], [], [], []]
    for action in actionList:
        rep = int(action.values[0, channel + 1])
        if rep == 0:        #暂时不考虑休息状态
            continue
        length = math.floor((len(action) - timeWindow) / strideWindow) + 1
        for j in range(length):
            subImage = action.iloc[strideWindow * j:strideWindow * j + timeWindow, 0:channel]  # 连续分割方式
            emgList[rep].append(subImage)
            labelList[rep].append(action.iloc[0, channel])

    for i in range(len(emgList)):
        emgList[i] = np.array(emgList[i])
    for i in range(len(labelList)):
        labelList[i] = np.array(labelList[i])
    return emgList, labelList


for j in range(1, 2):
    df = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/raw/DB2_s' + str(j) + 'raw.h5', 'df')

    '''滑动窗口分割'''
    actionList = action_seg(df, 12, 1)
    unList = uniform(actionList, 12)
    emgList, labelList = action_comb(actionList, 400, 100)
    # # 存储为h5文件
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/df_Seg/DB2_s' + str(j) + 'Seg.h5', 'w')
    file.create_dataset('Data1', data=(emgList[1]).astype('float32'))
    file.create_dataset('Data2', data=(emgList[2]).astype('float32'))
    file.create_dataset('Data3', data=(emgList[3]).astype('float32'))
    file.create_dataset('Data4', data=(emgList[4]).astype('float32'))
    file.create_dataset('Data5', data=(emgList[5]).astype('float32'))
    file.create_dataset('Data6', data=(emgList[6]).astype('float32'))
    file.create_dataset('label1', data=(labelList[1]).astype('int'))
    file.create_dataset('label2', data=(labelList[2]).astype('int'))
    file.create_dataset('label3', data=(labelList[3]).astype('int'))
    file.create_dataset('label4', data=(labelList[4]).astype('int'))
    file.create_dataset('label5', data=(labelList[5]).astype('int'))
    file.create_dataset('label6', data=(labelList[6]).astype('int'))
    file.close()
    print('******************DB2_s' + str(j) + '分割完成***********************')

