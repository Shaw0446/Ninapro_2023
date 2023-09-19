import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tfdeterminism import patch


#确定随机数
from Models.PopularModel.EnergyAndEmg import EnergyAndEmgCNN, EnergyAndEmgsoft

patch()
np.random.seed(123)
tf.random.set_seed(123)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"



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

if __name__ == '__main__':
    for j in range(1,2):
        file_emg = h5py.File('D:/Pengxiangdong/ZX/DB2/data/df_Seg/DB2_s' + str(j) + 'Seg.h5', 'r')
        emg1, emg3, emg4, emg6 = file_emg['x_train1'][:], file_emg['x_train3'][:]\
            , file_emg['x_train4'][:], file_emg['x_train6'][:]

        file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/energy_map/DB2_s' + str(j) + 'map.h5', 'r')
        # 数据集划分呢
        map1, map3, map4, map6 = file['map1'][:], file['map3'][:], file['map4'][:], file['map6'][:]
        y_train1, y_train3, y_train4, y_train6 = file['y_train1'][:], file['y_train3'][:], file['y_train4'][:], file['y_train6'][:]
        maptest = file['maptest'][:]
        y_test = file['y_test'][:]


        X_train = np.concatenate([emg1, emg3, emg4, emg6], axis=0)
        map_train = np.concatenate([map1, map3, map4, map6], axis=0)
        y_train = np.concatenate([y_train1, y_train3, y_train4, y_train6], axis=0)
        Y_train = nf.get_categorical(y_train)

        X_test = file_emg['x_test'][:]
        Y_test = nf.get_categorical(y_test)
        file_emg.close()
        file.close()
        print()

        callbacks = [#1设置学习率衰减,2保存最优模型
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath='D:/Pengxiangdong/ZX/DB2/DB2_model'
                                     '/DB2_s' + str(j) + 'model.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = EnergyAndEmgsoft()
        history = model.fit([map_train,X_train], Y_train, epochs=100, verbose=2, batch_size=64
            ,validation_data=([maptest,X_test],Y_test)
            , callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

