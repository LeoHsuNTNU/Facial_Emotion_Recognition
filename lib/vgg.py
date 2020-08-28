# -*- coding: utf-8 -*-
"""vgg.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17p9yhnAyXXsjbBZyzNySMOZHJAa6NZwA
"""

# import the self-define module
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('{0}/lib'.format(current_path))
import tensorflow as tf
import numpy as np
import weight
import math
import ops
import network

class VGG16(tf.keras.Model):
  def __init__(self, target_num, dropout_rate, name):
    super(VGG16, self).__init__()
    #self.input_data_shape = input_data_shape
    self.DNN_name = name
    self.target_num = target_num
    self.dropout_rate = dropout_rate
    #self.load_weights = load_weights

    # Construct the initial data of VGG16 network.
    self.conv1_1 = network.conv2d(64, [3, 3], [1, 1, 1, 1])

    self.conv1_2 = network.conv2d(64, [3, 3], [1, 1, 1, 1])

    self.conv2_1 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.conv2_2 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.conv3_1 = network.conv2d(256, [3, 3], [1, 1, 1, 1])

    self.conv3_2 = network.conv2d(256, [3, 3], [1, 1, 1, 1])

    self.conv3_3 = network.conv2d(256, [3, 3], [1, 1, 1, 1])

    self.conv4_1 = network.conv2d(512, [3, 3], [1, 1, 1, 1])

    self.conv4_2 = network.conv2d(512, [3, 3], [1, 1, 1, 1])

    self.conv4_3 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.conv5_1 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.conv5_2 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.conv5_3 = network.conv2d(128, [3, 3], [1, 1, 1, 1])

    self.full1 = network.fully_connected(4096)

    self.full2 = network.fully_connected(4096)

    self.full3 = network.fully_connected(self.target_num)

      
      
      
    # For batch normalization
    self.bn1 = network.batch_norm()

    self.bn2 = network.batch_norm()
    
    self.bn3 = network.batch_norm()

    self.bn4 = network.batch_norm()

    self.bn5 = network.batch_norm()

    self.bn6 = network.batch_norm()

    self.bn7 = network.batch_norm()

    self.bn8 = network.batch_norm()

    self.bn9 = network.batch_norm()

    self.bn10 = network.batch_norm()

    self.bn11 = network.batch_norm()

    self.bn12 = network.batch_norm()

    self.bn13 = network.batch_norm()

    self.bn14 = network.batch_norm()

    self.bn15 = network.batch_norm()


  def call(self, inputs, training=None):
    is_training = training
    if is_training:
      dropout_rate = tf.constant(self.dropout_rate, dtype=tf.float32)
    else:
      dropout_rate = tf.constant(0, dtype=tf.float32)
    
    #tf.keras.Input(shape=self.input_data_shape)
    with tf.name_scope('{0}/conv_block_1'.format(self.DNN_name)):
      conv1 = self.conv1_1(inputs)
      conv1 = self.bn1(conv1, is_training=is_training)
      relu1 = tf.nn.relu(conv1)
      
      conv2 = self.conv1_2(relu1)
      conv2 = self.bn2(conv2, is_training=is_training)
      relu2 = tf.nn.relu(conv2)

      max_pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool_0')


    with tf.name_scope('{0}/conv_block_2'.format(self.DNN_name)):
      conv3 = self.conv2_1(max_pool1)
      conv3 = self.bn3(conv3, is_training=is_training)
      relu3 = tf.nn.relu(conv3)

      conv4 = self.conv2_2(relu3)
      conv4 = self.bn4(conv4, is_training=is_training)
      relu4 = tf.nn.relu(conv4)
      
      max_pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
      
    with tf.name_scope('{0}/conv_block_3'.format(self.DNN_name)):
      conv5 = self.conv3_1(max_pool2)
      conv5 = self.bn5(conv5, is_training=is_training)
      relu5 = tf.nn.relu(conv5)
 
      conv6 = self.conv3_2(relu5)
      conv6 = self.bn6(conv6, is_training=is_training)
      relu6 = tf.nn.relu(conv6)

      conv7 = self.conv3_3(relu6)
      conv7 = self.bn7(conv7, is_training=is_training)
      relu7 = tf.nn.relu(conv7)
      
      max_pool3 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
      
      
    with tf.name_scope('{0}/conv_block_4'.format(self.DNN_name)):
      conv8 = self.conv4_1(max_pool3)
      conv8 = self.bn8(conv8, is_training=is_training)
      relu8 = tf.nn.relu(conv8)
      
      conv9 = self.conv4_2(relu8)
      conv9 = self.bn9(conv9, is_training=is_training)
      relu9 = tf.nn.relu(conv9)

      
      conv10 = self.conv4_3(relu9)
      conv10 = self.bn10(conv10, is_training=is_training)
      relu10 = tf.nn.relu(conv10)

      max_pool4 = tf.nn.max_pool(relu10, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
      
      
    with tf.name_scope('{0}/conv_block_5'.format(self.DNN_name)):
      conv11 = self.conv5_1(max_pool4)
      conv11 = self.bn11(conv11, is_training=is_training)
      relu11 = tf.nn.relu(conv11)
      
      conv12 = self.conv5_2(relu11)
      conv12 = self.bn12(conv12, is_training=is_training)
      relu12 = tf.nn.relu(conv12)
      
      conv13 = self.conv5_3(relu12)
      conv13 = self.bn13(conv13, is_training=is_training)
      relu13 = tf.nn.relu(conv13)
    
    with tf.name_scope('{0}/fully_connected_network'.format(self.DNN_name)):
      # flatten
      flatten_output = network.flatten(relu13)          
          
      full1 = self.full1(flatten_output)
      full1 = self.bn14(full1, is_training=is_training)
      relu14 = tf.nn.relu(full1)
      relu14 = tf.nn.dropout(tf.nn.relu(relu14), dropout_rate)
          
          
      full2 = self.full2(relu14)
      full2 = self.bn15(full2, is_training=is_training)
      relu15 = tf.nn.relu(full2)
      relu15 = tf.nn.dropout(tf.nn.relu(relu15), dropout_rate)
          
      full3 = self.full3(relu15)
      
      return full3