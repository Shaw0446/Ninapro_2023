import tensorflow as tf
from tensorflow.keras.layers import *

def downstage_resBlock(x, stage_id, keep_prob, stage_num=5):
    """
    Vnet左侧的压缩路径的一个stage层
    :param x: 该stage的输入
    :param stage_id: int,表示第几个stage，原论文中从上到下依次是1-5
    :param keep_prob: dropout保留元素的概率，如果不需要则设置为1
    :param stage_num: stage_num是Vnet设置的stage总数
    :return: stage下采样后的输出和stage下采样前的输出，下采样前的输出需要与Vnet右侧的扩展路径连接，所以需要输出保存。
    """
    x0 = x  # x0是stage的原始输入
    # Vnet每个stage的输入会进行特定次数的卷积操作，1~3个stage分别执行1~3次卷积，3以后的stage均执行3次卷积
    # 每个stage的通道数(卷积核个数或叫做feature map数量)依次增加两倍，从16，32，64，128到256
    for _ in range(3 if stage_id > 3 else stage_id):
        x = Conv2D(16 * (2 ** (stage_id - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        print('conv_down_stage_%d:' % stage_id, x.get_shape().as_list())  # 输出收缩路径中每个stage内的卷积
    x_add = ReLU()(add([x0, x]))
    x_add = Dropout(keep_prob)(x_add)

    if stage_id < stage_num:
        x_downsample = Conv2D(16 * (2 ** stage_id), 2, strides=(2, 2), activation=None, padding='same',kernel_initializer='he_normal')(x_add)
        x_downsample= BatchNormalization()(x_downsample)
        x_downsample = ReLU()(x_downsample)
        return x_downsample, x_add  # 返回每个stage下采样后的结果,以及在相加之后的结果
    else:
        return x_add, x_add  # 返回相加之后的结果，为了和上面输出保持一致，所以重复输出


def upstage_resBlock(forward_x, x, stage_id):
    """
    Vnet右侧的扩展路径的一个stage层
    :param forward_x: 对应压缩路径stage层下采样前的特征，与当前stage的输入进行叠加(不是相加)，补充压缩损失的特征信息
    :param x: 当前stage的输入
    :param stage_id: 当前stage的序号，右侧stage的序号和左侧是一样的，从下至上是5到1
    :return:当前stage上采样后的输出
    """
    input = concatenate([forward_x, x], axis=-1)
    for _ in range(3 if stage_id > 3 else stage_id):
        input = Conv2D(16 * (2 ** (stage_id - 1)), 5, activation=None, padding='same', kernel_initializer='he_normal')(input)
        input = BatchNormalization()(input)
        input = ReLU()(input)
        print('conv_down_stage_%d:' % stage_id, x.get_shape().as_list())  # 输出收缩路径中每个stage内的卷积
    conv_add = ReLU()(add([x, input]))
    if stage_id > 1:
        # 上采样的卷积也称为反卷积，或者叫转置卷积
        conv_upsample=Conv2DTranspose(16 * (2 ** (stage_id - 2)), 2, strides=(2, 2), padding='valid', activation=None,
                            kernel_initializer='he_normal')(conv_add)
        conv_upsample =BatchNormalization()(conv_upsample)
        conv_upsample = ReLU()(conv_upsample)
        return conv_upsample
    else:
        return conv_add


#
# def Vnet(pretrained_weights=None, input_size=(256, 256, 1), num_class=1, is_training=True, stage_num=5):
#     """
#     Vnet网络构建
#     :param pretrained_weights:是否加载预训练参数
#     :param input_size: 输入图像尺寸(w,h,c),c是通道数
#     :param num_class:  数据集的类别总数
#     :param is_training:  是否是训练模式
#     :param stage_num:  Vnet的网络深度，即stage的总数，论文中为5
#     :return: Vnet网络模型
#     """
#     keep_prob = 0.5 if is_training else 1.0  # dropout概率
#     left_featuremaps = []
#     input_data = Input(input_size)
#     x = PReLU()(BatchNormalization()(
#         Conv2D(16, 5, activation=None, padding='same', kernel_initializer='he_normal')(input_data)))
#
#     # 数据经过Vnet左侧压缩路径处理
#     for s in range(1, stage_num + 1):
#         x, featuremap = downstage_resBlock(x, s, keep_prob, stage_num)
#         left_featuremaps.append(featuremap)  # 记录左侧每个stage下采样前的特征
#
#     # Vnet左侧路径跑完后，需要进行一次上采样(反卷积)
#     x_up = PReLU()(BatchNormalization()(
#         Conv2DTranspose(16 * (2 ** (s - 2)), 2, strides=(2, 2), padding='valid', activation=None,
#                         kernel_initializer='he_normal')(x)))
#
#     # 数据经过Vnet右侧扩展路径处理
#     for d in range(stage_num - 1, 0, -1):
#         x_up = upstage_resBlock(left_featuremaps[d - 1], x_up, d)
#     if num_class > 1:
#         conv_out = Conv2D(num_class, 1, activation='softmax', padding='same', kernel_initializer='he_normal')(x_up)
#     else:
#         conv_out = Conv2D(num_class, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(x_up)
#
#     model = K.Model(inputs=input_data, outputs=conv_out)
#     print(model.output_shape)
#
#     if num_class > 1:
#         model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, momentum=0.99, decay=1e-6), loss='sparse_categorical_crossentropy',
#                       metrics=['ce'])  # metrics看看需不需要修改
#     else:
#         model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, momentum=0.99, decay=1e-6), loss='binary_crossentropy',
#                       metrics=['binary_accuracy'])
#     if pretrained_weights:
#         model.load_weights(pretrained_weights)
#     # plot_model(model, to_file='model.png')  # 绘制网络结构
#     return model
