import h5py
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.signal as signal
import matplotlib.pyplot as plt
import nina_funcs as nf

from Util.function import get_twoSet

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
# gestures = list(range(1,50))


def fea_transpose(data):
    # 为了便于归一化，对矩阵进行转置
    arr_T = np.transpose(np.array(data))
    sc_fea = preprocessing.StandardScaler().fit_transform(arr_T)
    arr_fea = np.transpose(sc_fea)
    return arr_fea

root_data = 'D:/Pengxiangdong/ZX/'
if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File(root_data + 'DB2/data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        file.close()
        emg_train, emg_test, label_train, label_test = get_twoSet(emg, label, rep)

        '''step2: 选择特征组合和归一化'''
        # 选择特征组合
        # frequency_features = ["fr", "mnp", "mnf", "mdf", "pkf"]

        fea_train = nf.frequency_features_extractor(shape=(emg_train.shape[0], -1), data=emg_train)
        fea_test = nf.frequency_features_extractor(shape=(emg_test.shape[0], -1), data=emg_test)

        fea_all = np.concatenate([fea_train, fea_test], axis=0)
        #
        ss = preprocessing.StandardScaler()
        ss.fit(fea_train)
        sc_train = ss.transform(fea_train)
        sc_test = ss.transform(fea_test)

        # fea_all = np.concatenate([sc_train, sc_test], axis=0)
        # 存储为h5文件
        file = h5py.File(root_data + 'DB2/data/stimulus/Fea/DB2_s' + str(j) + '15fea.h5', 'w')
        file.create_dataset('fea_all', data=fea_all.astype('float32'))
        file.create_dataset('fea_label', data=label.astype('int'))
        file.create_dataset('fea_rep', data=rep.astype('int'))
        file.close()
        print('******************DB2_s' + str(j) + '分割完成***********************')