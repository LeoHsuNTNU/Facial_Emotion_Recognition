import tensorflow as tf
import numpy as np

class Weight():
  def __init__(self):
    pass
  def conv_weight(self, conv_weight_name,in_channel, out_channel, kernel_width, kernel_height):
    self.conv_weight_name = conv_weight_name
    self.in_channel = in_channel
    self.out_channel = out_channel
    self.kernel_width = kernel_width
    self.kernel_height = kernel_height

    conv_weight = tf.Variable(tf.random.truncated_normal([self.kernel_height, self.kernel_width, self.in_channel,
                                                   self.out_channel], stddev=0.1, dtype=tf.float32), name='{0}'.format(self.conv_weight_name), trainable=True)
    return(conv_weight)

  def fully_weight(self, fully_weight_name,inchannel, outchannel):
    self.fully_weight_name = fully_weight_name
    self.inchannel = inchannel
    self.outchannel = outchannel

    fully_weight = tf.Variable(tf.random.truncated_normal([self.inchannel, self.outchannel],
                          stddev=0.1, dtype=tf.float32, name='{0}'.format(self.fully_weight_name)))
    return(fully_weight)
  
  def bias(self, bias_weight_name, shape, ini_type):
    self.bias_weight_name = bias_weight_name
    self.shape = shape
    self.ini_type = ini_type

    if self.ini_type == 'zeros':
      bias = tf.Variable(tf.zeros([self.shape], dtype=tf.float32), name='{0}'.format(self.bias_weight_name))
      return(bias)
    elif self.ini_type == 'truncated_normal':
      bias =  tf.Variable(tf.random.truncated_normal([self.shape], stddev=0.1, dtype=tf.float32))
      return(bias)
    else:
      print('Error! ini_type: zeros or truncated_normal. Please choose one')