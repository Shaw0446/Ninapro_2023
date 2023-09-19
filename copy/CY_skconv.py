# -*- coding: utf-8 -*-
import h5py
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow_core.python.keras import Input, Model

from Models.TrainModel.MutiCTrain import pltCurve


class SKNet(Model):

  def __init__(self, num_class):
    super(SKNet, self).__init__()

    self.num_class = num_class

    # Block 1
    self.conv1_a = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="SAME")
    self.bn1_a = layers.BatchNormalization()
    self.conv1_b = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding="SAME")
    self.bn1_b = layers.BatchNormalization()

    self.fc1 = layers.Dense(8, activation=None)
    self.bn1_fc = layers.BatchNormalization()

    self.fc1_a = layers.Dense(16, activation=None)

    # Block 2
    self.conv2_a = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="SAME")
    self.bn2_a = layers.BatchNormalization()
    self.conv2_b = layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="SAME")
    self.bn2_b = layers.BatchNormalization()

    self.fc2 = layers.Dense(16, activation=None)
    self.bn2_fc = layers.BatchNormalization()

    self.fc2_a = layers.Dense(32, activation=None)

    # Block 2
    self.conv3_a = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME")
    self.bn3_a = layers.BatchNormalization()
    self.conv3_b = layers.Conv2D(filters=64, kernel_size=5, strides=1, padding="SAME")
    self.bn3_b = layers.BatchNormalization()

    self.fc3 = layers.Dense(32, activation=None)
    self.bn3_fc = layers.BatchNormalization()

    self.fc3_a = layers.Dense(64, activation=None)

    # FC
    self.fc_out = layers.Dense(self.num_class, activation=None)

    self.maxpool = layers.MaxPool2D(pool_size=(2, 2))

  def call(self, x, training=False, verbose=False):

    if (verbose): print(x.shape)

    """ Conv-1 """
    # Split-1
    u1_a = tf.keras.activations.relu(self.bn1_a(self.conv1_a(x), training=training))
    u1_b = tf.keras.activations.relu(self.bn1_b(self.conv1_b(x), training=training))

    # Fuse-1
    u1 = u1_a + u1_b
    s1 = tf.math.reduce_sum(u1, axis=(1, 2))
    z1 = tf.keras.activations.relu(self.bn1_fc(self.fc1(s1), training=training))

    # Select-1
    a1 = tf.keras.activations.softmax(self.fc1_a(z1))
    a1 = tf.expand_dims(a1, 1)
    a1 = tf.expand_dims(a1, 1)
    b1 = 1 - a1
    v1 = (u1_a * a1) + (u1_b * b1)
    if (verbose): print(v1.shape)
    p1 = self.maxpool(v1)
    if (verbose): print(p1.shape)

    """ Conv-2 """
    # Split-2
    u2_a = tf.keras.activations.relu(self.bn2_a(self.conv2_a(p1), training=training))
    u2_b = tf.keras.activations.relu(self.bn2_b(self.conv2_b(p1), training=training))

    # Fuse-2
    u2 = u2_a + u2_b
    s2 = tf.math.reduce_sum(u2, axis=(1, 2))
    z2 = tf.keras.activations.relu(self.bn2_fc(self.fc2(s2), training=training))

    # Select-2
    a2 = tf.keras.activations.softmax(self.fc2_a(z2))
    a2 = tf.expand_dims(a2, 1)
    a2 = tf.expand_dims(a2, 1)
    b2 = 1 - a2
    v2 = (u2_a * a2) + (u2_b * b2)
    if (verbose): print(v2.shape)
    p2 = self.maxpool(v2)
    if (verbose): print(p2.shape)

    """ Conv-3 """
    # Split-3
    u3_a = tf.keras.activations.relu(self.bn3_a(self.conv3_a(p2), training=training))
    u3_b = tf.keras.activations.relu(self.bn3_b(self.conv3_b(p2), training=training))

    # Fuse-3
    u3 = u3_a + u3_b
    s3 = tf.math.reduce_sum(u3, axis=(1, 2))
    z3 = tf.keras.activations.relu(self.bn3_fc(self.fc3(s3), training=training))

    # Select-3
    a3 = tf.keras.activations.softmax(self.fc3_a(z3))
    a3 = tf.expand_dims(a3, 1)
    a3 = tf.expand_dims(a3, 1)
    b3 = 1 - a3
    v3 = (u3_a * a3) + (u3_b * b3)
    if (verbose): print(v3.shape)

    gap = tf.math.reduce_sum(v3, axis=(1, 2))
    if (verbose): print(gap.shape)
    out = self.fc_out(gap)
    if (verbose): print(out.shape)

    return out


if __name__ == '__main__':
  # inputs = Input([None, None, 32])
  # x = SKConv(3, G=1)(inputs)
  #
  # m = Model(inputs, x)
  # m.summary()

  file = h5py.File('../../data/DB2_S1segment2e4.h5', 'r')
  X_train = file['trainData'][:]
  Y_train = file['trainLabel'][:]
  X_test = file['testData'][:]
  Y_test = file['testLabel'][:]

  Y_train = tf.keras.utils.to_categorical(np.array(Y_train))
  Y_test = tf.keras.utils.to_categorical(np.array(Y_test))

  model = SKNet(49)
  history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, Y_test))

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  pltCurve(loss, val_loss, accuracy, val_accuracy)
  # model.save('MutiCNN-100E2e4.h5')
  tf.keras.utils.plot_model(model, to_file='../ModelPng/MutiCNN.png', show_shapes=True)
