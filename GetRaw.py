import h5py
import scipy.io as scio
import numpy as np
import nina_funcs as nf
import pandas as pd

dir='D:/Pengxiangdong/ZX/DB2/Ninapro'

if __name__ == '__main__':
    for j in range(1, 41):
        df1 = nf.get_redata(dir + '\DB2_s' + str(j), 'S' + str(j) + '_E1_A1.mat')
        df2 = nf.get_redata(dir + '\DB2_s' + str(j), 'S' + str(j) + '_E2_A1.mat')
        df3 = nf.get_redata(dir + '\DB2_s' + str(j), 'S' + str(j) + '_E3_A1.mat')

        dfall = pd.concat([df1, df2, df3],ignore_index=True)
        # 考虑数据放大到1
        dfmax = dfall.copy(deep=True)
        dfmax.iloc[:, :12] = dfmax.iloc[:, :12]
        df = dfmax.astype(np.float32)

        df.to_hdf('D:/Pengxiangdong/ZX/DB2/data/restimulus/reraw/DB2_s' + str(j) + 'raw.h5',format='table',key='df', mode='w',complevel=9, complib='blosc')

        print('******************DB2_s' + str(j) + '读取完成***********************')

        # file = h5py.File('F:/DB2/raw/DB2_s' + str(j) + 'raw.h5', 'w')
        # file.create_dataset('alldata', data=(dfall).astype('float32'))
        # file.close()











