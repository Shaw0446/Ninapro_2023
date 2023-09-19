import os
import tensorflow as tf
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import Model, Input
from tensorflow_core.python.keras.layers import Bidirectional, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout, \
    Dense, LSTM


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def MutiCNN():
    classes=49
    X_input = tf.keras.layers.Input(input_shape=(400,12))
    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(20, 1), strides=(10, 1), padding='same', activation='relu')(X_input)
    X = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(5, 1))(X)

    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(15, 1), strides=(6, 1), padding='same', activation='relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(10, 1))(X)

    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(15, 1), strides=(4, 1), padding='same',activation='relu')(X)
    X = tf.keras.layers.Flatten()(X)
    output1 = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(5, 1))(X)

    #第二层
    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(20, 1), strides=(1, 1), padding='same',activation='relu')(X_input)
    X = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(10, 1))(X)

    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(20, 1), strides=(5, 1), padding='same',activation='relu')(X)
    X = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(10, 1))(X)

    X = tf.keras.layers.Conv2DConv2D(filters=64, kernel_size=(20, 1), strides=(5, 1), padding='same',activation='relu')(X)
    output2 = tf.keras.layers.MaxPooling2D(pool_size=(10, 1), strides=(10, 1))(X)


    #第三层
    X = tf.keras.layers.Conv2D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu',
                               name='conv1')(X_input)
    X = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, name='pool3')(X)

    X = tf.keras.layers.Flatten(name='flatten')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(128, activation='relu', name='fc1')(X)
    X = tf.keras.layers.Dropout(0.5)(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', name='fc2')(X)
    mymodel = tf.keras.Model(inputs=X_input, outputs=X, name='CNN')
    mymodel.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

    return mymodel

def NewModel():

    I_input = Input(shape=(400, 12))
    X_input = tf.expand_dims(input=I_input,axis=3)
    classes = 49
    f1 = [20, 15, 10, 5]
    f2 = [10, 8, 6, 4]
    concats = []

    # 循环CNN支路
    for i in range(3):
        x = Conv2D(filters=32, kernel_size=(f1[i], 3), strides=(1, 1), activation='relu', padding='same')(X_input)
        x = MaxPooling2D((20, 1))(x)

        x = Conv2D(filters=64, kernel_size=(f2[i], 1), strides=(1, 1), activation='relu', padding='same')(x)
        x = MaxPooling2D((9 - 2 - i, 1))(x)

        x = Flatten()(x)
        concats.append(x)

    # LSTM网络支路
    # parameters for LSTM
    nb_lstm_outputs = 32  # 神经元个数
    nb_time_steps = 28  # 时间序列长度
    nb_input_vector = 28  # 输入序列
    l1 = Bidirectional(LSTM(units=nb_lstm_outputs, return_sequences=True), merge_mode='concat')(I_input)
    l1 = Flatten()(l1)
    concats.append(l1)

    merge = concatenate(concats, axis=1)
    X = merge
    X = Dropout(0.5)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax')(X)
    model = Model(inputs=I_input, outputs=X)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
    file = h5py.File('../../data/DB2_S1segment2e4.h5', 'r')
    X_train = file['trainData'][:]
    Y_train = file['trainLabel'][:]
    X_test = file['testData'][:]
    Y_test = file['testLabel'][:]

    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)

    Y_train = tf.keras.utils.to_categorical(np.array(Y_train))
    Y_test = tf.keras.utils.to_categorical(np.array(Y_test))

    model = NewModel()
    history = model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1
                        ,validation_data=(X_test, Y_test))
    #
    # loss= history.history['loss']
    # val_loss = history.history['val_loss']
    # accuracy =history.history['accuracy']
    # val_accuracy =history.history['val_accuracy']
    # pltCurve(loss, val_loss, accuracy, val_accuracy)
    model.save('MutiCLSTM-50E2e4.h5')
    tf.keras.utils.plot_model(model, to_file='../ModelPng/MutiCLSTM.png',show_shapes=True)



