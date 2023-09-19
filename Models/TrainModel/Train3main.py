import tensorflow as tf
import h5py
import numpy as np
import nina_funcs as nf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from Models.DBEmgNet.Away3CBAMNEW import Away3reluBNCBAMcatNEW
from Util.SepData import Sep3Data
from tfdeterminism import patch


#确定随机数
from Util.function import get_threeSet

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
        file = h5py.File('D:/Pengxiangdong/ZX/DB2/data/stimulus/TimeSeg/DB2_s' + str(j) + 'Seg.h5', 'r')
        # 数据集划分呢
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        file.close()

        emg_train, emg_vali, emg_test, label_train, label_vail, label_test = get_threeSet(emg, label, rep,6)
        Y_train = nf.get_categorical(label_train)
        Y_vali = nf.get_categorical(label_vail)

        Xtrain1, Xtrain2, Xtrain3 = Sep3Data(emg_train)
        Xvali1, Xvali2, Xvali3 = Sep3Data(emg_vali)


        callbacks = [#1设置学习率衰减,2保存最优模型
            # ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath='D:/Pengxiangdong/ZX/DB2/DB2_model/DB2_s' + str(j) + 'seg.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = Away3reluBNCBAMcatNEW()
        history = model.fit([Xtrain1, Xtrain2, Xtrain3], Y_train, epochs=50, verbose=2, batch_size=64
            ,validation_data=([Xvali1, Xvali2, Xvali3],Y_vali), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

