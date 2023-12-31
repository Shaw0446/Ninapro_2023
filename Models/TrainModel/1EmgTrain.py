import os

import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tfdeterminism import patch


#确定随机数
from Model1D.Away12CNN1D import Away12reluBNCNN1D
from Models.DBEmgNet.Away3CBAMNEW import Away3reluBNCBAMcatNEW
from Models.DBEmgNet.emgFile import reluBNCBAMcat
from Util.SepData import Sep3Data, Sep12Data
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
    for j in range(1, 2):
        file = h5py.File(root_data+'data/stimulus/Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        file.close()

        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_threeSet(emg, label, rep,6)
        Y_train = nf.get_categorical(label_train)
        Y_vail = nf.get_categorical(label_vail)

        file.close()

        callbacks = [#1设置学习率衰减,2保存最优模型
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath='D:/Pengxiangdong/ZX/DB2/DB2_model/emgmodel/DB2_s' + str(j) + 'testmodel.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model =reluBNCBAMcat()
        model.summary()
        history = model.fit(emg_train, Y_train, epochs=50, verbose=2, batch_size=64
                            # , callbacks=callbacks)
                            , validation_data=(emg_vali, Y_vail), callbacks=callbacks)
