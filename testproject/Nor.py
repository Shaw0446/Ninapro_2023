import h5py
import pandas as pd
import numpy as np
import scipy.signal as signal
import nina_funcs as nf


train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))




for j in range(1, 2):
    df = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/stimulus/raw/DB2_s' + str(j) + 'raw.h5', 'df')

    df1 = nf.normalise(df.copy(deep=True), train_reps)

    df2 = df1.copy(deep=True)

    df2.to_hdf('D:/Pengxiangdong/ZX/DB2/data/stimulus/raw/DB2_s' + str(j) + 'raw2.h5', format='table', key='df', mode='w',
              complevel=9, complib='blosc')

    print('******************DB2_s' + str(j) + '分割完成***********************')



