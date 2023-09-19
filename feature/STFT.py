import biosppy
import h5py
import scipy
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

from Preprocess.Segment import actionSegment


def stft(sig, **params):
    '''
    快速傅里叶变换时时频域图像
    :param sig: 输入信号
    :param params: {fs:采样频率；
                    window:窗。默认为汉明窗；
                    nperseg： 每个段的长度，默认为256，
                    noverlap:重叠的点数。指定值时需要满足COLA约束。默认是窗长的一半，
                    nfft：fft长度，
                    detrend：（str、function或False）指定如何去趋势，默认为Flase，不去趋势。
                    return_onesided：默认为True，返回单边谱。
                    boundary：默认在时间序列两端添加0
                    padded：是否对时间序列进行填充0（当长度不够的时候），
                    axis：可以不必关心这个参数}
    :return: f:采样频率数组；t:段时间数组；Zxx:STFT结果
    Zxx返回的是二维数组，每一列数据某一时间段的频谱，每一行表示某一频率的不同时间的复数。
    '''
    f, t, zxx = signal.stft(sig, **params)
    return f, t, zxx

def stft_specgram(sig, picname=None, **params):    #picname是给图像的名字，为了保存图像
    f, t, zxx = stft(sig, fs=2000, nperseg=128)
    plt.pcolormesh(t, f, np.abs(zxx), shading='auto')
    plt.colorbar()
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()
    # if picname is not None:
    #     plt.savefig( str(picname) + '.jpg')       #保存图像
    # plt.clf()      #清除画布
    return f, t, zxx



if __name__ == '__main__':
    for j in range(1, 2):
        h5 = h5py.File('F:/DB2/raw/DB2_s' + str(j) + 'raw.h5', 'r')
        alldata = h5['alldata'][:]
        seglist = actionSegment(alldata, 1, 12)
        # bnlist = bnsegment(seglist)i
        stftarr=[]
        for k in range(6):
            iemg = seglist[k].data
            for m in range(12):
                f, t, zxx=signal.spectrogram(iemg[0:400, m],fs=2000,noverlap=128)
                stftarr.append(f)
            print()




