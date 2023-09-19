import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL



''' 消融实验---多流批标准化'''
def Away3reluBConv():
    input1 = KL.Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = KL.Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)

    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    output1 =x1

    x2 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input21)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    output2 = x2

    x3 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input31)
    x3 = KL.BatchNormalization()(x3)
    x3 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x3)
    x3 = KL.BatchNormalization()(x3)
    output3 = x3

    c = KL.Concatenate(axis=-2)([output1, output2, output3])
    X = KL.GlobalAvgPool2D()(c)
    X = KL.Dense(512, activation='relu')(X)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Away3reluBConv2():
    input1 = KL.Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = KL.Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)

    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    output1 = x1

    x2 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input21)
    x2 = KL.Activation('relu')(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1),  padding='same')(x2)
    x2 = KL.BatchNormalization()(x2)
    x2 = KL.Activation('relu')(x2)
    output2 = x2

    x3 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input31)
    x3 = KL.BatchNormalization()(x3)
    x3 = KL.Activation('relu')(x3)
    x3 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1),  padding='same')(x3)
    x3 = KL.BatchNormalization()(x3)
    x3 = KL.Activation('relu')(x3)
    output3 = x3

    c = KL.Concatenate(axis=-2)([output1, output2, output3])
    X = KL.GlobalAvgPool2D()(c)
    X = KL.Dense(128, activation='relu')(X)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model





''' 消融实验---单流批标准化'''
def SinglereluBN():
    input1 = KL.Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = KL.Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)
    #早期融合网络，加入1×1卷积
    c1 = KL.Concatenate(axis=-2)([input11, input21, input31])


    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(c1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    output1 =x1


    X = KL.GlobalAvgPool2D()(output1)
    X = KL.Dense(512, activation='relu')(X)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


''' 消融实验---单流'''
def Singlerelu():
    input1 = KL.Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = KL.Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)
    #早期融合网络，加入1×1卷积
    c1 = KL.Concatenate(axis=-2)([input11, input21, input31])


    x1 = KL.Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(c1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    output1 =x1


    X = KL.GlobalAvgPool2D()(output1)
    X = KL.Dense(512, activation='relu')(X)
    X = KL.Dropout(0.1)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model