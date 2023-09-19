import h5py
import numpy as np
import wfdb
from scipy.signal import resample
from wfdb import processing
import pandas as pd
import nina_funcs as nf


def downsample(data, oldFS, newFS):
    """
    Resample data from oldFS to newFS using the scipy 'resample' function.

    Parameters
    ----------
    data : instance of pandas.core.DataFrame
        Data to resample.
    oldFS : float
        The sampling frequency of data.
    newFS : float
        The new sampling frequency.

    Returns:

    newData : instance of pandas.DataFrame
        The downsampled dataset.
    """

    newNumSamples = int((data.shape[0] / oldFS) * newFS)
    newData = pd.DataFrame(resample(data, newNumSamples))
    return newData


if __name__ == '__main__':
    for j in range(1,2):
        df = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/raw/DB2_s' + str(j) + 'raw.h5', 'df')
        data = np.array(df)
        channel = 12
        index = []
        tempemg = []
        templabel = []
        temprep = []
        #数据降采样
        for i in range(channel):
            index = processing.resample_sig(data[:, i], fs=2000, fs_target=100)
        downlabs = index[1]
        #标签降采样
        for m in downlabs:
            tempemg.append(data[int(m), :channel])

        for m in downlabs:
            templabel.append(data[int(m), channel])

        for m in downlabs:
            temprep.append(data[int(m), channel + 1])
        downemg = (np.array(tempemg))
        label = (np.array(templabel))
        rep = np.array(temprep)
        # 转为DataFrame存储
        df_down = pd.DataFrame(downemg)
        df_down['stimulus'] = label
        df_down['repetition'] = rep

    # 存储为h5文件
        df_down.to_hdf('D:/Pengxiangdong/ZX/DB2/data/df_down/DB2_s' + str(j) + 'down.h5', format='table', key='df', mode='w', complevel=9, complib='blosc')
        print()
