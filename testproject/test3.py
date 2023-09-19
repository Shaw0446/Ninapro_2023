import h5py
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import EMGImg
import seaborn as sns




if __name__ == '__main__':
    a = 1.00000234414423555563632
    d = float(a)
    b = np.array([a, a, a, a])
    c = b.astype('float32')

    print(b[0])
    print(c[0])
    print(d)
