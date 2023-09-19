import os

import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tfdeterminism import patch
from Models.DBFeaEmgNet.FeaEmgFile import FeaAndEmg_se, FeaAndEmg_model1, FeaAndEmg_modelsoft
from Util.function import get_threeSet
#确定随机数

patch()
np.random.seed(123)
tf.random.set_seed(123)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.compat.v1.ConfigProto()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))



def pltCurve(loss, val_loss, accuracy, val_accuracy):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation val_accuracy')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

root_data = 'D:/Pengxiangdong/ZX/'

if __name__ == '__main__':
    for j in (1, 26, 38):
        file = h5py.File(root_data+'DB2/data/stimulus/Seg/DB2_s' + str(j) + 'SegEnhance.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_threeSet(emg, label, rep, 6)
        file.close()
        Y_train = nf.get_categorical(label_train)
        Y_vali = nf.get_categorical(label_vail)


        feaFile = h5py.File(root_data + 'DB2/data/stimulus/Fea/DB2_s' + str(j) + 'feaEnhance.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_vali, fea_test, feay_train, feay_vali, feay_test = get_threeSet(fea_all, fea_label, fea_rep, 6)


        '''特征向量数据适应'''
        fea_train = np.expand_dims(fea_train, axis=-1)
        fea_vali = np.expand_dims(fea_vali, axis=-1)

        callbacks = [  # 1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath=root_data+'DB2/DB2_model2/ab/134-6/DB2_s' + str(j) + 'emgfeamodel.h5'
                            , monitor='val_accuracy', save_best_only=True)]
        model = FeaAndEmg_modelsoft()
        # model.summary()
        history = model.fit([emg_train, fea_train], Y_train, epochs=50, verbose=2, batch_size=64
                            # , callbacks=callbacks)
                            , validation_data=([emg_vali, fea_vali], Y_vali), callbacks=callbacks)

