import h5py
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import nina_funcs as nf




if __name__ == '__main__':
    for j in range(1, 2):
        df1 = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/raw/DB2_s' + str(j) + 'raw.h5', 'df')
        dfemg_band = nf.filter_data(data=df1, f=(10, 500), butterworth_order=4, btype='bandpass')
        dfemg_notch = nf.notch_filter(data=dfemg_band, f0=50, Q=30, fs=2000)
        # 存储为h5文件
        dfemg_notch.to_hdf('D:/Pengxiangdong/ZX/DB2/data/filter/DB2_s' + str(j) + 'filter.h5', format='table',key='df', mode='w', complevel=9, complib='blosc')

        # file = h5py.File('F:/DB2/filter/DB2_s' + str(j) + 'filter.h5', 'w')
        # file.create_dataset('alldata', data=(alldata).astype('float32'))
        # file.close()