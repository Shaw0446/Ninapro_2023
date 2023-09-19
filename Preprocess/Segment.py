import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import nina_funcs as nf


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))




for j in range(1, 2):
    df = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/stimulus/filter/DB2_s' + str(j) + 'filter.h5', 'df')

    '''step2: 滑动窗口分割'''
    emg, label, rep = nf.windowing(df, reps=[], gestures=gestures, win_len=400, win_stride=100)
    # x_test, y_test, r_test = nf.windowing(df, reps=test_reps, gestures=gestures, win_len=20, win_stride=1)

    # 存储为h5文件
    file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/stimulus/TimeSeg/DB2_s' + str(j) + 'Seg.h5', 'w')
    file.create_dataset('emg', data=emg.astype('float32'))
    file.create_dataset('label', data=label.astype('int32'))
    file.create_dataset('rep', data=rep.astype('int32'))
    file.close()

    print('******************DB2_s' + str(j) + '分割完成***********************')


