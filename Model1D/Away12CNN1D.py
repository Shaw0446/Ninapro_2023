import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL

def BConv(layer):
    x = KL.Conv1D(filters=64, kernel_size=8, strides=1, padding='same')(layer)
    x = KL.Activation('relu')(x)
    x = KL.BatchNormalization()(x)
    x = KL.Conv1D(filters=128, kernel_size=2, strides=1, padding='same')(x)
    x = KL.Activation('relu')(x)
    x = KL.BatchNormalization()(x)
    return x



def Away12reluBNCNN1D():
    input1 = KL.Input(shape=(400, 1))
    input2 = KL.Input(shape=(400, 1))
    input3 = KL.Input(shape=(400, 1))
    input4 = KL.Input(shape=(400, 1))
    input5 = KL.Input(shape=(400, 1))
    input6 = KL.Input(shape=(400, 1))
    input7 = KL.Input(shape=(400, 1))
    input8 = KL.Input(shape=(400, 1))
    input9 = KL.Input(shape=(400, 1))
    input10 = KL.Input(shape=(400, 1))
    input11 = KL.Input(shape=(400, 1))
    input12 = KL.Input(shape=(400, 1))

    output1 = BConv(input1)
    output2 = BConv(input2)
    output3 = BConv(input3)
    output4 = BConv(input4)
    output5 = BConv(input5)
    output6 = BConv(input6)
    output7 = BConv(input7)
    output8 = BConv(input8)
    output9 = BConv(input9)
    output10 = BConv(input10)
    output11 = BConv(input11)
    output12 = BConv(input12)

    c = KL.Concatenate(axis=-1)([output1, output2,output3, output4,output5, output6,output7, output8,output9, output10,output11,output12])
    X = KL.GlobalAvgPool1D()(c)
    X = KL.Dense(128, activation='relu')(X)
    X = KL.Dropout(0.5)(X)
    s = KL.Dense(49, activation='softmax')(X)
    model = tf.keras.Model(inputs=[input1, input2,input3, input4,input5, input6,input7, input8,input9, input10,input11,input12], outputs=s)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
