from copy import deepcopy
import h5py
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
from Models.TrainModel import Train3main
from Util.SepData import Sep3Data, Sep12Data


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



if __name__ == '__main__':
    for j in range(1, 2):
        file = h5py.File('F:/DB2/DownSegZsc/DB2_s' + str(j) + 'allDownZsc.h5', 'r')

        Data0, Data1, Data2, Data3, Data4, Data5 = file['Data0'][:], file['Data1'][:], file['Data2'][:] \
            , file['Data3'][:], file['Data4'][:], file['Data5'][:]
        Label0, Label1, Label2, Label3, Label4, Label5 = file['label0'][:], file['label1'][:], file['label2'][:] \
            , file['label3'][:], file['label4'][:], file['label5'][:]
        file.close()
        # 选定手势做训练集，测试集，验证集
        # X_test=Data4
        # Y_test=Data4

        X_test = np.concatenate([Data1, Data4], axis=0)
        Y_test = np.concatenate([Label1, Label4], axis=0)

        Xtest1, Xtest2, Xtest3,Xtest4, Xtest5, Xtest6,Xtest7, Xtest8, Xtest9,Xtest10, Xtest11, Xtest12 = Sep12Data(X_test)

        model = keras.models.load_model('F:/DB2/model/DownAway12reluBNCNN1D/111'
                                     '/DB2_s'+ str(j) + '6seg205mZsc.h5')

        Y_test = to_categorical(np.array(Y_test))
        Y_predict = model.predict([Xtest1, Xtest2, Xtest3,Xtest4, Xtest5, Xtest6,Xtest7, Xtest8, Xtest9,Xtest10, Xtest11, Xtest12])

        # # 返回每行中概率最大的元素的列坐标（热编码转为普通标签）
        y_pred = Y_predict.argmax(axis=1)
        y_true = Y_test.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred)
        # plot_confusion_matrix(cm,'1C-50E-2e4.png')
        classes = []
        for i in range(len(cm)):
            classes.append(str(i))
        contexts = classification_report(y_true, y_pred, target_names=classes, digits=4)

        with open("F:/DB2/result/111/DB2_s"+str(j)+"6seg205mZsc.txt", "w", encoding='utf-8') as f:
            f.write(str(contexts))
            f.close()
        # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


