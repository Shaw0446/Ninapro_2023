"""
Clean and simple Keras implementation of SE block as described in:
    - [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

Python 3.
"""

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Reshape, Permute, multiply


def SEBlock(se_ratio=16, activation="relu", data_format='channels_last', ki="he_normal"):
    '''
    se_ratio : ratio for reduce the filter number of first Dense layer(fc layer) in block
    activation : activation function that of first dense layer
    data_format : channel axis is at the first of dimension or the last
    ki : kernel initializer
    '''

    def f(input_x):
        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = input_x._keras_shape[channel_axis]

        reduced_channels = input_channels // se_ratio

        # Squeeze operation
        x = GlobalAveragePooling2D()(input_x)
        x = Reshape(1, 1, input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(reduced_channels, kernel_initializer=ki)(x)
        x = Activation(activation)(x)
        # Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = Permute(dims=(3, 1, 2))(x) if data_format == 'channels_first' else x
        x = multiply([input_x, x])

        return x

    return f