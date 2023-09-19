import h5py
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField, GramianAngularField
import cv2 as cv
import numpy as np
import numpy.fft as fft


def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


for j in range(1, 2):
    h5 = h5py.File('F:/DB2/Downfilter/DB2_s' + str(j) + 'down.h5', 'r')
    alldata = h5['alldata'][:]
    # 动作状态数据分割  肌电子图标准化
    temp = (alldata[0:20,0]).reshape(-1,1)
    mtf = MarkovTransitionField(image_size=1)
    X_mtf = mtf.fit_transform(temp)
    print()