import os

import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tfdeterminism import patch


#确定随机数
from Models.DBEmgNet.Away3CBAMNEW import Away3reluBNCBAMcatNEW
from Models.DBEmgNet.emgFile import reluBNCBAMcat
from Models.DBFeaNet.FeaModel import Fea_select, Fea_select2
from Util.SepData import Sep3Data
from Util.function import get_threeSet

patch()
np.random.seed(123)
tf.random.set_seed(123)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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




root_data = 'D:/Pengxiangdong/ZX/DB2/'

if __name__ == '__main__':
    for j in range(1, 26, 38):
        feaFile = h5py.File(root_data + 'data/stimulus/Fea/DB2_s' + str(j) + 'fea9.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_vali, fea_test, feay_train, feay_vali, feay_test = get_threeSet(fea_all, fea_label, fea_rep, 6)

        Y_train = nf.get_categorical(feay_train)
        Y_vali = nf.get_categorical(feay_vali)

        '''特征向量数据适应'''
        fea_train = np.expand_dims(fea_train, axis=-1)
        fea_vali = np.expand_dims(fea_vali, axis=-1)

        callbacks = [#1设置学习率衰减,2保存最优模型
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
                              cooldown=0, min_lr=0),
            ModelCheckpoint(filepath='D:/Pengxiangdong/ZX/DB2/DB2_model2/ab/DB2_s' + str(j) + 'fea_select_model.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model =Fea_select2()
        model.summary()
        history = model.fit(fea_train, Y_train, epochs=50, verbose=2, batch_size=64
                            , validation_data=(fea_vali, Y_vali ), callbacks=callbacks)
