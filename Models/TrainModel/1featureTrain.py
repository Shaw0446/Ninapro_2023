import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import nina_funcs as nf
from Models.DBFeaNet.FeaModel import model3
from tfdeterminism import patch

#确定随机数
from Util.function import get_threeSet

patch()
np.random.seed(123)
tf.random.set_seed(123)



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
    root_data = 'D:/Pengxiangdong/ZX/'
    for j in range(1,2):
        feaFile = h5py.File(root_data + 'DB2/data/stimulus/Fea/DB2_s' + str(j) + 'fea.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_vali, fea_test, feay_train, feay_vali, feay_test = get_threeSet(fea_all, fea_label, fea_rep, 6)

        Y_train = nf.get_categorical(feay_train)
        Y_vali = nf.get_categorical(feay_vali)

        callbacks = [#1设置学习率衰减,2保存最优模型
            # EarlyStopping(monitor='val_accuracy', patience=5),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001,
            #                   cooldown=0, min_lr=0),
            ModelCheckpoint(filepath='D:/Pengxiangdong/ZX/DB2/DB2_model/DB2_s' + str(j) + 'Fea.h5'
            , monitor='val_accuracy', save_best_only=True)]
        model = model3()
        history = model.fit(fea_train , Y_train, epochs=50, verbose=2, batch_size=64
                            , validation_data=(fea_vali, Y_vali), callbacks=callbacks)

        # loss= history.history['loss']
        # val_loss = history.history['val_loss']
        # accuracy =history.history['accuracy']
        # val_accuracy =history.history['val_accuracy']32
        # pltCurve(loss, val_loss, accuracy, val_accuracy)
        #早停机制保存最优模型后，不再另外保存
        # model.save('D:/Pengxiangdong/ZX/modelsave/3Channel/Away3CBAMcat/DB2_s'+str(j)+'re25seg400100mZsc.h5')
        # tf.keras.utils.plot_model(model, to_file='../ModelPng/Away3reluBNCBAMcatNEW.png', show_shapes=True)

