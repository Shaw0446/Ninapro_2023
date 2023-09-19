import tensorflow as tf
from keras import regularizers
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import *

from Models.DBlayers.CBAM import cbam_acquisition, cbam_time
from Models.DBlayers.SENet import se_block
from Models.DBlayers.daNet import danet_resnet101
from Models.PopularModel.Vnet import downstage_resBlock, upstage_resBlock


def FeaAndEmg_model1():
    input1 = Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = Input(shape=(108, 1))

    # 早期融合网络，加入1×1卷积
    x1 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input11)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = cbam_time(x1)
    x1 = Conv2D(filters=128, kernel_size=(1, 12), strides=(1, 2), padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = cbam_acquisition(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(128)(x1)
    x1 = GlobalAvgPool2D()(x1)

    x2 = Conv1D(filters=32, kernel_size=6, strides=1, padding='same')(input2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = LocallyConnected1D(32, 1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(256,  activity_regularizer=regularizers.l1(0.01))(x2)
    x2 = GlobalAvgPool1D()(x2)


    X = Concatenate(axis=-1)([x1,x2])
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def FeaAndEmg_modelsoft():
    input1 = Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = Input(shape=(108, 1))

    # 早期融合网络，加入1×1卷积
    x1 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input11)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    # x1 = cbam_time(x1)
    x1 = Conv2D(filters=128, kernel_size=(1, 12), strides=(1, 1), padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    # x1 = cbam_acquisition(x1)
    x1 = GlobalAvgPool2D()(x1)
    x1 = KL.Dense(128, activation='relu')(x1)
    x1 = KL.Dropout(0.5)(x1)
    s1 = KL.Dense(49, activation='softmax', activity_regularizer=regularizers.l1(0.01))(x1)

    x2 = Conv1D(filters=32, kernel_size=6, strides=1, padding='same')(input2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = LocallyConnected1D(32, 1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = GlobalAvgPool1D()(x2)
    s2 = Dense(49, activation='softmax', activity_regularizer=regularizers.l1(0.01))(x2)

    s1 = s1
    s2 = s2
    s = Add()([s1, s2])

    model = tf.keras.Model(inputs=[input1,input2], outputs=s1)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def FeaAndEmg_model2():
    input1 = Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = Input(shape=(180, 1))

    # 早期融合网络，加入1×1卷积
    stage_num = 3
    left_featuremaps = []
    input_data = input11
    x = Conv2D(16, (5, 1), activation=None, padding='same', kernel_initializer='he_normal')(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 数据经过Vnet左侧压缩路径处理
    for s in range(1, stage_num + 1):
        x, featuremap = downstage_resBlock(x, s, 0.5, stage_num)
        left_featuremaps.append(featuremap)  # 记录左侧每个stage下采样前的特征

    # Vnet左侧路径跑完后，需要进行一次上采样(反卷积)
    x_up= Conv2DTranspose(16 * (2 ** (s - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                        kernel_initializer='he_normal')(x)
    x_up = BatchNormalization()(x_up)
    x_up = Activation('relu')(x_up)
    # 数据经过Vnet右侧扩展路径处理
    for d in range(stage_num - 1, 0, -1):
        x_up = upstage_resBlock(left_featuremaps[d - 1], x_up, d)
    x1 = Dense(128, activation='relu')(x_up)
    x1 = GlobalAvgPool2D()(x1)


    # x2 = Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input2)
    # x2 = MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    # x2 = Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(x2)
    # x2 = MaxPool1D(pool_size=2, strides=1, padding='same')(x2)

    x2 = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(input2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = LocallyConnected1D(32, 1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.5)(x2)

    x2 = GlobalAvgPool1D()(x2)


    X = Concatenate(axis=-1)([x1,x2])
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.1)(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1,input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model





def FeaAndEmg_se():
    input1 = KL.Input(shape=(400,12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input11 = KL.Reshape(target_shape=(20, 20, 12))(input11)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积

    x1 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    # x1 = se_block(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),  padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = KL.Activation('relu')(x1)
    # x1 = se_block(x1)
    x1 = KL.GlobalAvgPool2D()(x1)


    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='same')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='same')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)
    s = KL.Add()([x1, x2])

    X = KL.Dense(256, activation='relu')(s)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model