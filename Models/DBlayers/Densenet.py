import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
import tensorflow.keras.layers as KL


# model
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0,drop_rate=0.2):
    # Bottleneck layers
    # 1x1 Conv的作用是降低特征数量
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.LeakyReLU(alpha=alpha)(x)
    x = KL.Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # Composite function
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.LeakyReLU(alpha=alpha)(x)
    x = KL.Conv2D(nb_filter, (3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    if drop_rate: x = KL.Dropout(drop_rate)(x)

    return x


def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for nb in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = KL.concatenate([x, conv], axis=3)

    return x


def TransitionLayer(x, compression=0.5, alpha=0.0,  is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = KL.BatchNormalization(axis=3)(x)
    x = KL.LeakyReLU(alpha=alpha)(x)
    x = KL.Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = KL.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    else:
        x = KL.AveragePooling2D(pool_size=(5, 1), strides=(2, 1))(x)
    return x


def newDenseNet():
    input1 = KL.Input(shape=(400, 12))
    input11 = tf.expand_dims(input=input1, axis=3)
    growth_rate = 12

    x = KL.Conv2D(filters=64, kernel_size=(20, 4), strides=(1, 1), activation='relu', padding='same')(input11)
    x = KL.BatchNormalization()(x)
    x = DenseBlock(x, 6, growth_rate, drop_rate=0.5)
    x = TransitionLayer(x)
    x = DenseBlock(x, 6, growth_rate, drop_rate=0.5)
    x = TransitionLayer(x)
    x = DenseBlock(x, 6, growth_rate, drop_rate=0.5)
    x = KL.BatchNormalization(axis=3)(x)
    output1 = KL.GlobalAveragePooling2D()(x)

    s = KL.Dense(49, activation='softmax')(output1)
    model = tf.keras.Model(inputs=input1, outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


'''
    其他版本的实现方式，网络层次更深
'''
# def conv_block(x, nb_filter, dropout_rate=None, name=None):
#     inter_channel = nb_filter * 4
#
#     # 1x1 convolution
#     x = KL.BatchNormalization(epsilon=1.1e-5, axis=3, name=name + '_bn1')(x)
#     x = KL.Activation('relu', name=name + '_relu1')(x)
#     x = KL.Conv2D(inter_channel, 1, 1, name=name + '_conv1', use_bias=False)(x)
#
#     if dropout_rate:
#         x = KL.Dropout(dropout_rate)(x)
#
#     # 3x3 convolution
#     x = KL.BatchNormalization(epsilon=1.1e-5, axis=3, name=name + '_bn2')(x)
#     x = KL.Activation('relu', name=name + '_relu2')(x)
#     x = KL.ZeroPadding2D((1, 1), name=name + '_zeropadding2')(x)
#     x = KL.Conv2D(nb_filter, 3, 1, name=name + '_conv2', use_bias=False)(x)
#
#     if dropout_rate:
#         x = KL.Dropout(dropout_rate)(x)
#
#     return x
#
#
# def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
#                 grow_nb_filters=True, name=None):
#     concat_feat = x  # store the last layer output
#
#     for i in range(nb_layers):
#
#         branch = i + 1
#         x = conv_block(concat_feat, growth_rate, dropout_rate,
#                        name=name + str(stage) + '_block' + str(branch))  # 在参考的基础，修改的地方这里应该是相同的growth_rate=32
#         concat_feat = KL.Concatenate(axis=3, name=name + str(stage) + '_block' + str(branch))([concat_feat, x])
#
#         if grow_nb_filters:
#             nb_filter += growth_rate
#
#     return concat_feat, nb_filter
#
#
# def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, name=None):
#
#     x = KL.BatchNormalization(epsilon=1.1e-5, axis=3, name=name + str(stage) + '_bn')(x)
#     x = KL.Activation('relu', name=name + str(stage) + '_relu')(x)
#
#     x = KL.Conv2D(int(nb_filter * compression), 1, 1, name=name + str(stage) + '_conv', use_bias=False)(x)
#
#     if dropout_rate:
#         x = KL.Dropout(dropout_rate)(x)
#
#     x = KL.AveragePooling2D((2, 2), strides=(2, 2), name=name + str(stage) + '_pooling2d')(x)
#
#     return x
#
#
# def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
#              classes=1000, weights_path=None):
#     compression = 1.0 - reduction
#     nb_filter = 64
#     nb_layers = [6, 12, 24, 16]  # For DenseNet-121
#
#     img_input = tf.keras.Input(shape=(224, 224, 3))
#
#     # initial convolution
#     x = KL.ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
#     x = KL.Conv2D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)
#     x = KL.BatchNormalization(epsilon=1.1e-5, axis=3, name='conv1_bn')(x)
#     x = KL.Activation('relu', name='relu1')(x)
#     x = KL.ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
#     x = KL.MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
#
#     # Dense block and Transition layer
#     for block_id in range(nb_dense_block - 1):
#         stage = block_id + 2  # start from 2
#         x, nb_filter = dense_block(x, stage, nb_layers[block_id], nb_filter, growth_rate,
#                                    dropout_rate=dropout_rate, name='Dense')
#
#         x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, name='Trans')
#         nb_filter *= compression
#
#     final_stage = stage + 1
#     x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
#                                dropout_rate=dropout_rate, name='Dense')
#
#     # top layer
#     x = KL.BatchNormalization(name='final_conv_bn')(x)
#     x = KL.Activation('relu', name='final_act')(x)
#     x = KL.GlobalAveragePooling2D(name='final_pooling')(x)
#     x = KL.Dense(classes, activation='softmax', name='fc')(x)
#
#     model = Model(img_input, x, name='DenseNet121')
#
#     return model
