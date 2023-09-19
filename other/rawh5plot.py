import h5py
import matplotlib.pyplot as plt
import numpy as np
h5 = h5py.File('../data/DB2_S1raw.h5', 'r')
alldata = h5['alldata']
# h5.close()
data=np.array(alldata)
plt.plot(data[:,0:12])

plt.show()





