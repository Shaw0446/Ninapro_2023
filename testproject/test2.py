import math
import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import colors
import seaborn as sns
import EMGImg

# import matplotlib.colors as mcolors
# colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化
b = np.linspace(1, 400, 24)
c=[]
for sam in b:
    sam=int(sam)
    c.append(sam)
# 长片段热度图
for j in range(1, 2):
    h5 = h5py.File('F:/DB2/raw/DB2_s' + str(j) + 'raw.h5', 'r')
    alldata = h5['alldata'][:]
    temp = alldata[0:400, 0:12]
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    sns.set()
    ax = sns.heatmap(temp)
    ax.set_xticklabels(a)
    ax.set_yticklabels(c)
    plt.show()


#图像存储
# img = Image.open("E:/360MoveData/Users/Administrator/Desktop/1.jpg")
# img = np.array(img)
# print(img)
