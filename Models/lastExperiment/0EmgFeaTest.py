import h5py
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import nina_funcs as nf
from Models.DBlayers.CBAM import cbam_acquisition, cbam_time
from Util.function import get_twoSet

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.compat.v1.ConfigProto()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(20, 16), dpi=100)
    np.set_printoptions(precision=2)  # 用于控制Python中小数的显示精度
    classes = []
    for i in range(len(cm)):
        classes.append(i)
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))  # 显示对应的数字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    # show confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.colorbar()
    plt.savefig(savename, format='png')
    plt.show()


# 多分类中每个类别的评价标准
def multiclassEva(cm):
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    # 返回各个类的TP,TN,FP,FN   得到的数组可以通过np.mean()求整体的均值
    Pre = TP / (TP + FP)  # 查准率
    Recall = TP / (TP + FN)  # 查全率
    F1score = 2 * [(Pre * Recall) / (Pre + Recall)]

    return Recall, Pre, F1score

#整体分类的评价结果
def OverallEva(Recall, Pre, F1score):
    allRecall = np.mean(Recall)
    allF1score = np.mean(F1score)
    allPre = np.mean(Pre)
    return allRecall,  allPre, allF1score


root_data = 'D:/Pengxiangdong/ZX/'

if __name__ == '__main__':
    for j in (1, 26, 38):
        file = h5py.File(root_data+'DB2/data/stimulus/Seg/DB2_s' + str(j) + 'SegEnhance.h5', 'r')
        emg, label, rep = file['emg'][:], file['label'][:], file['rep'][:]
        emg_train,  emg_test, label_train, label_test = get_twoSet(emg, label, rep)
        file.close()


        feaFile = h5py.File(root_data + 'DB2/data/stimulus/Fea/DB2_s' + str(j) + 'feaEnhance.h5', 'r')
        # 将六次重复手势分开存储
        fea_all, fea_label, fea_rep = feaFile['fea_all'][:], feaFile['fea_label'][:], feaFile['fea_rep'][:]
        feaFile.close()
        fea_train, fea_test, feay_train, feay_test = get_twoSet(fea_all, fea_label, fea_rep)

        '''特征向量数据适应'''
        fea_test = np.expand_dims(fea_test, axis=-1)
        Y_test = nf.get_categorical(label_test)
        model = keras.models.load_model(root_data+'DB2/DB2_model2/ab/134-6/DB2_s' + str(j) + 'emgfeamodel.h5')


        Y_predict = model.predict([emg_test,fea_test])

        # # 返回每行中概率最大的元素的列坐标（热编码转为普通标签）
        y_pred = Y_predict.argmax(axis=1)
        y_true = Y_test.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred)
        # plot_confusion_matrix(cm,'1C-50E-2e4.png')
        classes = []
        for i in range(len(cm)):
            classes.append(str(i))
        contexts = classification_report(y_true, y_pred, target_names=classes, digits=4)

        with open(root_data+"DB2/result/Ablation/134-6-25/DB2_s"+str(j)+"emgfea.txt", "w", encoding='utf-8') as f:
            f.write(str(contexts))
            f.close()
        # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


