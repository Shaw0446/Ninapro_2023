import numpy
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# 特征级融合测试
def EnergyAndEmgCNN():
    input1 = KL.Input(shape=(60, 80))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 12))
    input21 = tf.expand_dims(input=input2, axis=3)

    #早期融合网络，加入1×1卷积

    x1 = KL.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1),  padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.LocallyConnected2D(64,kernel_size=(2,2))(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    f1 = KL.Flatten()(x1)


    x2 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input21)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.LocallyConnected2D(64 , kernel_size=(2,2))(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    f2 = KL.Flatten()(x2)


    #拼接
    s1 = KL.Dense(128)(f1)
    s1 = KL.Activation('relu')(s1)
    s1 = KL.BatchNormalization()(s1)

    s2 = KL.Concatenate()([f1,f2])
    s2 = KL.Dense(128)(s2)
    s2 = KL.Activation('relu')(s2)
    s2 = KL.BatchNormalization()(s2)
    s = KL.Concatenate()([s1,s2])
    s = KL.Dense(49, activation='softmax')(s)


    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
    #               loss={"first": 'categorical_crossentropy',
    #                     "second": 'categorical_crossentropy'},
    #               loss_weights={"first": 1,
    #                             "second": 1},
    #               metrics=['accuracy'])
    return model



# 决策层融合的测试,(显存资源不足)
def EnergyAndEmgsoft():
    input1 = KL.Input(shape=(60, 80))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 12))
    input21 = tf.expand_dims(input=input2, axis=3)

    #早期融合网络，加入1×1卷积

    x1 = KL.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1),  padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Dense(128, activation='relu')(x1)
    x1 = KL.Flatten()(x1)
    x1 = KL.Dropout(0.2)(x1)
    output1 = KL.Dense(49, activation='softmax')(x1)

    x2 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input21)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.LocallyConnected2D(64 , kernel_size=(2,2))(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.Flatten()(x2)
    x2 = KL.Dropout(0.2)(x2)
    output2 = KL.Dense(49, activation='softmax')(x2)

    #拼接

    s = KL.Add()([output1,output2])

    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
    #               loss={"first": 'categorical_crossentropy',
    #                     "second": 'categorical_crossentropy'},
    #               loss_weights={"first": 1,
    #                             "second": 1},
    #               metrics=['accuracy'])
    return model




