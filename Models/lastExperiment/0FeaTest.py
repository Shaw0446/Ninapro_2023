import os

import h5py
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import nina_funcs as nf
from Models.DBlayers.CBAM import cbam_acquisition, cbam_time
from Util.SepData import Sep3Data
from Util.function import get_twoSet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"



root_data = 'D:/Pengxiangdong/ZX/DB2/'
if __name__ == '__main__':
    for j in range(1, 2):
        feaFile = h5py.File(root_data + 'data/stimulus/Fea/DB2_s' + str(j) + 'fea6.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_test, feay_train, feay_test = get_twoSet(fea_all, fea_label, fea_rep)

        '''特征向量数据适应'''
        fea_test = np.expand_dims(fea_test, axis=-1)

        Y_test = nf.get_categorical(feay_test)
        model = keras.models.load_model(root_data+'DB2_model/DB2_s' + str(j) + 'fea_select_model.h5')
        Y_predict = model.predict(fea_test)

        # # 返回每行中概率最大的元素的列坐标（热编码转为普通标签）
        y_pred = Y_predict.argmax(axis=1)
        y_true = Y_test.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred)
        # plot_confusion_matrix(cm,'1C-50E-2e4.png')
        classes = []
        for i in range(len(cm)):
            classes.append(str(i))
        contexts = classification_report(y_true, y_pred, target_names=classes, digits=4)

        with open(root_data+"result/222/DB2_s"+str(j)+"fea6_select.txt", "w", encoding='utf-8') as f:
            f.write(str(contexts))
            f.close()
        # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


