import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#热度图
    # j=1
    # file = h5py.File('F:/DB2/resegment/DB2_s' + str(j) + 'reseg400100m.h5', 'r')
    # X_train = file['trainData'][:]
    # Y_train = file['trainLabel'][:]
    # X_test = file['testData'][:]
    # Y_test = file['testLabel'][:]
    #
    # sns.set()
    # img = X_train[0,:,:]
    # ax=sns.heatmap(img)
    # plt.show()

if __name__ == '__main__':

    input_shape=(4,10,10,3)
    x = np.array(range(1200),dtype=np.float64).reshape(input_shape)
    print(x.shape)
    y = tf.keras.layers.Conv2D(8, 5, strides=(2, 2), padding='same',data_format='channels_last',
                               )(x)
    print("卷积输出形状：", y.shape)











