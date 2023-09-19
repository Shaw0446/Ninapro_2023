import h5py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nina_funcs as nf

train_reps = [1, 3, 4, 6]
test_reps = [2, 5]
gestures = list(range(1, 50))


for j in range(1, 2):
    df = pd.read_hdf('D:/Pengxiangdong/ZX/DB2/data/raw/DB2_s' + str(j) + 'raw1.h5', 'df')
    df1 = nf.normalise(df.copy(deep=True), train_reps)
    df = np.array(df)
    df1=np.array(df1)
    plt.plot(df[12000:13000,0:12])