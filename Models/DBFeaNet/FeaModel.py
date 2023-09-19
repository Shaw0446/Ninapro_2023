import tensorflow as tf
from keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import *

from Models.DBlayers.CBAM import cbam_acquisition, cbam_time
from Models.PopularModel import Vnet
from Models.PopularModel.Vnet import downstage_resBlock, upstage_resBlock

channel_axis = 1 if K.image_data_format() == "channels_first" else 3





def Fea_select():

    input1 = Input(shape=(72, 1))

    x1 = Conv1D(filters=32, kernel_size=6, strides=1,  padding='same')(input1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    # x1 = Conv1D(filters=128, kernel_size=5, strides=1,  padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    x1 = LocallyConnected1D(32,1)(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(128)(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.5)(x1)
    x1 = GlobalAvgPool1D()(x1)

    s = Dense(49, activation='softmax',activity_regularizer=regularizers.l1(0.01))(x1)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def Fea_select2():

    input1 = Input(shape=(72, 1))

    x1 = Conv1D(filters=32, kernel_size=6, strides=1,  padding='same')(input1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    # x1 = Conv1D(filters=128, kernel_size=5, strides=1,  padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = Activation('relu')(x1)
    x1 = LocallyConnected1D(32,1)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(128, activation='relu')(x1)
    x1 = GlobalAvgPool1D()(x1)

    s = Dense(49, activation='softmax',activity_regularizer=regularizers.l1(0.01))(x1)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model









def Stage1():
    input00 = Input(shape=(20, 12))
    input0 = tf.expand_dims(input=input00, axis=3)
    input1 = Input(shape=(16, 16, 12))
    input2 = Input(shape=(16, 16, 12))

    x0 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input0)
    x0 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x0)
    x0 = LocallyConnected2D(64, (1, 1))(x0)
    x0 = LocallyConnected2D(64, (1, 1))(x0)
    x0 = Flatten()(x0)

    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x1 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = LocallyConnected2D(64, (1,1))(x1)
    x1 = LocallyConnected2D(64, (1,1))(x1)
    x1 = Flatten()(x1)

    x2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = LocallyConnected2D(64, (1, 1))(x2)
    x2 = LocallyConnected2D(64, (1, 1))(x2)
    x2 = Flatten()(x2)

    c = Add()([x1, x2])
    c = Concatenate()([c, x0])
    X = Dense(512, activation='relu')(c)
    X = Dense(256, activation='relu')(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input00, input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def Stage2():
    input1 = Input(shape=(16, 16, 12))
    input2 = Input(shape=(16, 16, 12))

    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x1 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = LocallyConnected2D(64, (1,1))(x1)
    x1 = LocallyConnected2D(64, (1,1))(x1)
    x1 = Dropout(0.1)(x1)
    x1 = Dense(256, activation='relu')(x1)

    x2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(input1)
    x2 = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = LocallyConnected2D(64, (1, 1))(x2)
    x2 = LocallyConnected2D(64, (1, 1))(x2)
    x2 = Dropout(0.1)(x2)
    x2 = Dense(256, activation='relu')(x2)

    c = Add()([x1, x2])
    X = Dense(256, activation='relu')(c)
    X = GlobalAvgPool2D()(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model3():
    input1 = Input(shape=(36))
    input2 = tf.expand_dims(input=input1, axis=-1)

    x1 = Conv1D(filters=64, kernel_size=2, strides=1, activation='relu', padding='same')(input2)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x1)
    x1 = Conv1D(filters=128, kernel_size=4, strides=1, activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)

    x1 = MaxPooling1D(pool_size=2, strides=1, padding='same')(x1)

    X = Flatten()(x1)

    # s1 = Dense(17, activation='softmax')(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    s2 = Dense(49, activation='softmax')(X)
    # s = Add()([s1,s2])

    model = tf.keras.Model(inputs=input1, outputs=s2)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model



