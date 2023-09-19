import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras import *

channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM 特征通道注意力
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])


# SAM 空间注意力
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(2, 2), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


'''
# 标准的CBAM，没有卷积！！！加入ResBlock'''
def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])




# 3条支路，每个支路做串行的卷积注意力,在网络层加入标准化，注意顺序,早期网络加入1×1卷积
def EnergyCNN():
    input1 = KL.Input(shape=(60, 80))
    input11 = tf.expand_dims(input=input1, axis=3)
    input11 = KL.BatchNormalization()(input11)


    #早期融合网络，加入1×1卷积
    x1 = KL.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l1(0.01))(input11)
    x1 = KL.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x1)
    x1 = KL.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l1(0.01))(x1)
    x1 = KL.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='same')(x1)
    x1 = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l1(0.01))(x1)
    x1 = KL.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x1)
    # x1 = cbam_module(x1)
    # x1 = KL.Conv1D(filters=64, kernel_size=20, strides=20, activation='relu', padding='valid')(input1)
    # x1 = KL.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)
    # x1 = KL.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid')(x1)
    # x1 = KL.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x1)

    c = KL.Flatten()(x1)
    X = KL.Dense(128, activation='relu')(c)
    X = KL.Dropout(0.5)(X)
    s = KL.Dense(17, activation='softmax')(X)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model



