import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import *

channel_axis = 1 if K.image_data_format() == "channels_first" else 3


# CAM 特征通道注意力
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = GlobalMaxPooling2D()(input_xs)
    maxpool_channel = Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = GlobalAvgPool2D()(input_xs)
    avgpool_channel = Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal',
                         use_bias=True, bias_initializer='zeros')
    Dense_Two = Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True,
                         bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = Activation('sigmoid')(channel_attention_feature)
    return Multiply()([channel_attention_feature, input_xs])


# # SAM 空间注意力
# def spatial_attention(channel_refined_feature):
#     maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
#     avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
#     max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
#     return Conv2D(filters=1, kernel_size=(2, 2), padding="same", activation='sigmoid',
#                      kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

# 时间注意力
def time_attention(channel_refined_feature):
    maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return Conv2D(filters=1, kernel_size=(2, 1), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


# Acquisition channel 采集通道注意力
def acquisition_attention(channel_refined_feature):
    maxpool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return Conv2D(filters=1, kernel_size=(1, 2), padding="same", activation='sigmoid',
                     kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

'''
# 标准的CBAM，没有卷积！！！加入ResBlock
def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = Multiply()([channel_refined_feature, spatial_attention_feature])
    return Add()([refined_feature, input_xs])
'''


# 加入ResBlock 采集通道注意力机制
def cbam_acquisition(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    acquisition_attention_feature = acquisition_attention(channel_refined_feature)
    refined_feature = Multiply()([channel_refined_feature, acquisition_attention_feature])
    return Add()([refined_feature, input_xs])
# 标准的CBAM，没有卷积！！！加入ResBlock
def cbam_time(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    time_attention_feature = time_attention(channel_refined_feature)
    refined_feature = Multiply()([channel_refined_feature, time_attention_feature])
    return Add()([refined_feature, input_xs])



# 3条支路，每个支路做串行的卷积注意力,在网络层加入标准化，注意顺序,早期网络加入1×1卷积
def Away3reluBNCBAMcatNEW():
    input1 = Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)
    #早期融合网络，加入1×1卷积
    c1 = Concatenate(axis=-2)([input11, input21, input31])
    c1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding='same')(c1)


    x1 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), padding='same')(input11)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = cbam_time(x1)
    x1 = Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1),  padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = BatchNormalization()(x1)
    output1 = cbam_acquisition(x1)


    x2 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input21)
    x2 = Activation('relu')(x2)
    x2 = BatchNormalization()(x2)
    x2 = cbam_time(x2)
    x2 = Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1),  padding='same')(x2)
    x2 = Activation('relu')(x2)
    x2 = BatchNormalization()(x2)
    output2 = cbam_acquisition(x2)


    x3 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1),  padding='same')(input31)
    x3 = Activation('relu')(x3)
    x3 = BatchNormalization()(x3)
    x3 = cbam_time(x3)
    x3 = Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1),  padding='same')(x3)
    x3 = Activation('relu')(x3)
    x3 = BatchNormalization()(x3)
    output3 = cbam_acquisition(x3)


    c2 = Concatenate(axis=-2)([output1, output2, output3])
    c = Concatenate(axis=-1)([c1, c2])
    X = GlobalAvgPool2D()(c)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.1)(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model




def Away3reluBNCBAMsoft():
    input1 = Input(shape=(400, 8))
    input11 = tf.expand_dims(input=input1, axis=3)
    input2 = Input(shape=(400, 2))
    input21 = tf.expand_dims(input=input2, axis=3)
    input3 = Input(shape=(400, 2))
    input31 = tf.expand_dims(input=input3, axis=3)
    #早期融合网络，加入1×1卷积
    c1 = Concatenate(axis=-2)([input11, input21, input31])
    c1 = Conv2D(filters=32, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(c1)


    x1 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input11)
    x1 = BatchNormalization()(x1)
    x1 = cbam_time(x1)
    x1 = Conv2D(filters=128, kernel_size=(1, 8), strides=(1, 1), activation='relu', padding='same')(x1)
    x1 = BatchNormalization()(x1)
    output1 = cbam_acquisition(x1)


    x2 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input21)
    x2 = BatchNormalization()(x2)
    x2 = cbam_time(x2)
    x2 = Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    output2 = cbam_acquisition(x2)


    x3 = Conv2D(filters=64, kernel_size=(8, 1), strides=(1, 1), activation='relu', padding='same')(input31)
    x3 = BatchNormalization()(x3)
    x3 = cbam_time(x3)
    x3 = Conv2D(filters=128, kernel_size=(1, 2), strides=(1, 1), activation='relu', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    output3 = cbam_acquisition(x3)


    c2 = Concatenate(axis=-2)([output1, output2, output3])
    c = Concatenate(axis=-1)([c1, c2])
    X = GlobalAvgPool2D()(c)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.1)(X)
    s = Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


