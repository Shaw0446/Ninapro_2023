from copy import deepcopy
import h5py
from tensorflow import keras
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt

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
    file = h5py.File('../../data/DB2_S1seg400mMinMax.h5', 'r')
    X_test = file['testData'][:]
    Y_test = file['testLabel'][:]
    model = keras.models.load_model('../TrainModel/savemodel/1Cmodel/1C-100E-seg400mMinMax.h5')
    Y_test = to_categorical(np.array(Y_test))
    Y_predict = model.predict(X_test)

    # # 返回每行中概率最大的元素的列坐标（热编码转为普通标签）
    y_pred = Y_predict.argmax(axis=1)
    y_true = Y_test.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(cm,'1C-50E-2e4.png')
    classes = []
    for i in range(len(cm)):
        classes.append(str(i))
    contexts = classification_report(y_true, y_pred, target_names=classes, digits=4)
    with open("1C-100Ebn-seg400mMinMax.txt", "w", encoding='utf-8') as f:
        f.write(str(contexts))
        f.close()
    # print(classification_report(y_true, y_pred, target_names=classes, digits=4))


