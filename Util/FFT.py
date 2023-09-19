import h5py
import numpy as np#导入一个数据处理模块
import pylab as plt#导入一个绘图模块，matplotlib下的模块
from numpy.fft import fft
import seaborn as sns


# 傅里叶变换频谱  横轴频率纵轴能量
def plot_FFT(signal):
    signal = signal
    iSampleCount = signal.shape[0]  # 采样数
    iSampleRate = 2000
    yf1 = np.abs(np.fft.rfft(signal) / iSampleCount)  # 快速傅里叶变换
    xf1 = np.linspace(0, iSampleRate / 2, int(iSampleCount / 2) + 1) #由于对称性，只取一半区间
    plt.figure(figsize=(12, 8), dpi=100)
    plt.subplot(211)
    plt.plot(signal)
    plt.title('Original wave')
    plt.subplot(212)
    plt.plot(xf1,yf1)
    plt.xlabel("Freq(Hz)")
    plt.title('FFT of Mixed wave')
    plt.show()

def plot_OnlyFFT(sig):
    sig=sig
    iSampleCount = sig.shape[0]  # 采样数
    iSampleRate = 2000
    yf1 = np.abs(np.fft.rfft(sig) / iSampleCount)  # 快速傅里叶变换
    xf1 = np.linspace(0, iSampleRate / 2, int(iSampleCount / 2) + 1)  # 由于对称性，只取一半区间
    plt.figure(figsize=(12, 8), dpi=100)
    plt.plot(xf1, yf1)
    plt.xlabel("Freq(Hz)")
    plt.title('FFT of Mixed wave')
    plt.show()

#以秒为单位
def get_plotsigfre(sig,title = None):
    iSampleRate = 2000  # 采样频率
    iSampleCount = sig.shape[0]  # 采样数
    t = np.linspace(0, iSampleCount / iSampleRate, iSampleCount)
    xFFT = np.abs(np.fft.rfft(sig) / iSampleCount)  # 快速傅里叶变换
    xFreqs = np.linspace(0, iSampleRate / 2, int(iSampleCount / 2) + 1)  ##对称性只取一半
    plt.figure(figsize=(12, 8), dpi=100)

    ax0 = plt.subplot(211)  # 画时域信号
    plt.title(title)
    ax0.set_xlabel("Time(s)")
    # plt.xlim(0, 800)
    ax0.set_ylabel("Amp(μV)")
    ax0.plot(t, sig)

    ax1 = plt.subplot(212)  # 画频域信号-振幅谱  xFreqs为频率， xFFT为能量
    ax1.set_xlabel("Freq(Hz)")
    ax1.set_ylabel("Power")
    ax1.plot(xFreqs, xFFT)
    sns.set()
    plt.show()





if __name__ == '__main__':
    for j in range(1,2):
        h5 = h5py.File('F:/DB2/reraw/DB2_s' + str(j) + 'reraw.h5', 'r')
        alldata = h5['alldata']
        plot_FFT(alldata[12000:12400,1])