import h5py
import matplotlib.pyplot as plt
import numpy as np



def get_plotsigfre(sig,title = None):
    start= 0
    end=len(sig)
    iSampleRate = 2000  # 采样频率
    signal = sig[start:end]
    iSampleCount = signal.shape[0]  # 采样数
    t = np.linspace(0, iSampleCount / iSampleRate, iSampleCount)

    xFFT = np.abs(np.fft.rfft(signal) / iSampleCount)  # 快速傅里叶变换
    xFreqs = np.linspace(0, iSampleRate / 2, int(iSampleCount / 2) + 1)

    plt.figure(figsize=(10, 6))

    ax0 = plt.subplot(211)  # 画时域信号
    plt.title(title)
    ax0.set_xlabel("Time(s)")
    # plt.xlim(0, 800)
    ax0.set_ylabel("Amp(μV)")
    ax0.plot(t, signal)

    ax1 = plt.subplot(212)  # 画频域信号-频谱
    ax1.set_xlabel("Freq(Hz)")
    ax1.set_ylabel("Power")
    ax1.plot(xFreqs, xFFT)
    plt.show()

if __name__ == '__main__':
    h5 = h5py.File('../data/DB2_S1raw.h5', 'r')
    alldata = h5['alldata']
    emg=alldata[:,0]
    get_plotsigfre(emg)



