import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import signal
import os
import pywt
from tqdm import tqdm
from sklearn.decomposition import PCA
import scipy as sp


# DWPT分解
def emg_dwpt(signal, wavelet_name='db1'):
    wavelet_level = int(np.log2(len(signal)))
    wp = pywt.WaveletPacket(signal, wavelet_name, mode='sym')
    coeffs = []
    level_coeff = wp.get_level(wavelet_level)
    for i in range(len(level_coeff)):
        coeffs.append(level_coeff[i].data)
    coeffs = np.array(coeffs)
    coeffs = coeffs.flatten()
    return coeffs


''' ---频域特征分隔线--- '''


def fft(data):
    return np.fft.fft(data)


# (样本为20时，输出20)
def psd(data):
    return np.abs(np.fft.fft(data)) ** 2

# Mean frequency
def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den

# Median frequency
def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]


''' ---时域特征分隔线--- '''




def iemg(data):
    return np.sum(np.abs(data))

def mav(data):
    signal_abs = [abs(s) for s in data]
    signal_abs = np.array(signal_abs)
    if len(signal_abs) == 0:
        return 0
    else:
        return np.mean(signal_abs)

def wl(data):
    return np.sum(abs(np.diff(data)))

def var(data):
    return np.var(data)

def rms(data):
    return np.sqrt(np.mean(data ** 2))


# (20)
def hist(data, nbins=20):
    histsig, bin_edges = np.histogram(data, bins=nbins)
    return tuple(histsig)


# (1)
def entropy(data):
    pk = sp.stats.rv_histogram(np.histogram(data, bins=20)).pdf(data)
    return sp.stats.entropy(pk)


# (1)
def kurtosis(data):
    return sp.stats.kurtosis(data)


# np.diff后一个元素减去前一个，np.sign返回由1和-1组成的数组
def zero_cross(data):
    return len(np.where(np.diff(np.sign(data)))[0])


def min(data):
    return np.min(data)


def max(data):
    return np.max(data)


def mean(data):
    return np.mean(data)


def median(data):
    return np.median(data)

def emg_ssc(signal, threshold=1e-5):
    signal = np.array(signal)
    temp = [(signal[i] - signal[i - 1]) * (signal[i] - signal[i + 1]) for i in range(1, signal.shape[0] - 1, 1)]
    temp = np.array(temp)

    temp = temp[temp >= threshold]
    return temp.shape[0]


def emg_hemg(signal, bins):
    signal = np.array(signal)
    hist, bin_edge = np.histogram(signal, bins)
    return hist




def frequency_features_estimation(signal, fs, frame, step):
    """
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size
    :return: frequency_features_matrix: narray matrix with the frequency features stacked by columns.
    """

    fr = []
    mnp = []
    mnf = []
    mdf = []
    pkf = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(frequency_ratio(frequency, power))  # Frequency ratio
        mnp.append(np.sum(power) / len(power))  # Mean power
        mnf.append(mean_freq(frequency, power))  # Mean frequency
        mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency

    frequency_features_matrix = np.column_stack((fr, mnp, mnf, mdf, pkf))

    return frequency_features_matrix


def frequency_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC



def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()