from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL


def se_block(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    se_feature = GlobalAveragePooling2D()(input_feature)
    se_feature = Reshape((1, 1, channel))(se_feature)  # 第一步：压缩(Squeeze), reshape成1✖️1✖️C
    # assert se_feature._keras_shape[1:] == (1,1,channel)
    # 第二步：激励(Excitation),
    # 由两个全连接层组成，其中SERatio是一个缩放参数，这个参数的目的是为了减少通道个数从而降低计算量。
    # 第一个全连接层有(C/radio)个神经元，输入为1×1×C，输出1×1×(C/radio)。
    # 第二个全连接层有C个神经元，输入为1×1×(C/radio)，输出为1×1×C。
    se_feature = Dense(channel // ratio,
                       activation='relu',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    # assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
    se_feature = Dense(channel,
                       activation='sigmoid',
                       kernel_initializer='he_normal',
                       use_bias=True,
                       bias_initializer='zeros')(se_feature)
    # assert se_feature._keras_shape[1:] == (1, 1, channel)
    """
    # 因为keras默认为channels_last,没修改不需要加这段
    if K.image_data_format() == 'channels_first':
        se_feature = Permute((3, 1, 2))(se_feature)
    """
    se_feature = multiply([input_feature, se_feature])
    return se_feature


def SENet_model():
    input1 = KL.Input(shape=(400, 12))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = KL.Input(shape=(36, 1))

    # 早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=64, kernel_size=(10, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = KL.BatchNormalization()(x1)
    x1 = se_block(x1)
    x1 = KL.Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = KL.BatchNormalization()(x1)
    x1 = se_block(x1)
    x1 = KL.GlobalAvgPool2D()(x1)


    x2 = KL.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(input2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.Conv1D(filters=128, kernel_size=5, strides=1, activation='relu', padding='valid')(x2)
    x2 = KL.MaxPool1D(pool_size=2, strides=1, padding='valid')(x2)
    x2 = KL.GlobalAvgPool1D()(x2)

    X = KL.Concatenate(axis=-1)([x1, x2])
    X = KL.Dense(256, activation='relu')(X)
    X = KL.Dropout(0.2)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
