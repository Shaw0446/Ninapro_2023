import math
import h5py
import pandas as pd
import numpy as np



def get_threeSet(emg, label, rep_arr, rep_vali):
    train_reps = [1, 3, 4, 6]
    test_reps = [2, 5]
    #从训练集剔除验证集
    train_reps.remove(rep_vali)
    x = [np.where(rep_arr[:] == rep) for rep in test_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    emg_test = emg[indices, :]
    label_test = label[indices]
    rep_test=rep_arr[indices]
    x = [np.where(rep_arr[:] == rep_vali)]
    indices2 = np.squeeze(np.concatenate(x, axis=-1))
    emg_vali = emg[indices2, :]
    label_vali = label[indices2]
    rep_vali=rep_arr[indices2]
    x = [np.where(rep_arr[:] == rep) for rep in train_reps]
    indices3 = np.squeeze(np.concatenate(x, axis=-1))
    emg_train = emg[indices3, :]
    label_train = label[indices3]
    rep_train=rep_arr[indices3]

    return emg_train, emg_vali, emg_test, label_train, label_vali, label_test


def get_twoSet(emg, label, rep_arr):
    train_reps = [1, 3, 4, 6]
    test_reps = [2, 5]
    x = [np.where(rep_arr[:] == rep) for rep in test_reps]
    indices = np.squeeze(np.concatenate(x, axis=-1))
    emg_test = emg[indices, :]
    label_test = label[indices]
    x = [np.where(rep_arr[:] == rep) for rep in train_reps]
    indices3 = np.squeeze(np.concatenate(x, axis=-1))
    emg_train = emg[indices3, :]
    label_train = label[indices3]

    return emg_train, emg_test, label_train, label_test

# if __name__ == '__main__':
#     train_reps = [1, 3, 4, 6]
#     test_reps = 3
#     train_reps.remove(test_reps)
#     train_reps
