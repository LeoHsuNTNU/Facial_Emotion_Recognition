# import the self-define module
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('{0}/lib'.format(current_path))
import tensorflow as tf
import numpy as np
import weight
import math

class VGG16(tf.Module):
  def __init__(self, input_data_shape, DNN_name, target_num, load_weights, dropout, dropout_rate=None):
    self.input_data_shape = input_data_shape
    #self.data_label = data_label
    self.DNN_name = DNN_name
    self.target_num = target_num
    self.load_weights = load_weights
    self.dropout = dropout
    self.dropout_rate = dropout_rate
    
    image_height = self.input_data_shape[1]
    image_width = self.input_data_shape[2]
    num_channels = self.input_data_shape[3]
    print(image_height, image_width, num_channels)
    Weight = weight.Weight()
    resulting_width = math.ceil(image_width / (2 * 2 * 2 * 2))
    resulting_height = math.ceil(image_height / (2 * 2 * 2 * 2))

    full1_input_size = resulting_width * resulting_height * 512
    if self.load_weights == 'no':
      with tf.name_scope('{0}'.format(self.DNN_name)):
        self.conv1_weight = Weight.conv_weight('conv1_weight', num_channels, 64, 3, 3)
        self.conv1_bias = Weight.bias('conv1_bias', 64, 'zeros')

        self.conv2_weight = Weight.conv_weight('conv2_weight', 64, 64, 3, 3)
        self.conv2_bias = Weight.bias('conv2_bias', 64, 'zeros')

        self.conv3_weight = Weight.conv_weight('conv3_weight', 64, 128, 3, 3)
        self.conv3_bias = Weight.bias('conv3_bias', 128, 'zeros')

        self.conv4_weight = Weight.conv_weight('conv4_weight', 128, 128, 3, 3)
        self.conv4_bias = Weight.bias('conv4_bias', 128, 'zeros')

        self.conv5_weight = Weight.conv_weight('conv5_weight', 128, 256, 3, 3)
        self.conv5_bias = Weight.bias('conv5_bias', 256, 'zeros')

        self.conv6_weight = Weight.conv_weight('conv6_weight', 256, 256, 3, 3)
        self.conv6_bias = Weight.bias('conv6_bias', 256, 'zeros')

        self.conv7_weight = Weight.conv_weight('conv7_weight', 256, 256, 3, 3)
        self.conv7_bias = Weight.bias('conv7_bias', 256, 'zeros')

        self.conv8_weight = Weight.conv_weight('conv8_weight', 256, 512, 3, 3)
        self.conv8_bias = Weight.bias('conv8_bias', 512, 'zeros')

        self.conv9_weight = Weight.conv_weight('conv9_weight', 512, 512, 3, 3)
        self.conv9_bias = Weight.bias('conv9_bias', 512, 'zeros')

        self.conv10_weight = Weight.conv_weight('conv10_weight', 512, 512, 3, 3)
        self.conv10_bias = Weight.bias('conv10_bias', 512, 'zeros')

        self.conv11_weight = Weight.conv_weight('conv11_weight', 512, 512, 3, 3)
        self.conv11_bias = Weight.bias('conv11_bias', 512, 'zeros')

        self.conv12_weight = Weight.conv_weight('conv12_weight', 512, 512, 3, 3)
        self.conv12_bias = Weight.bias('conv12_bias', 512, 'zeros')

        self.conv13_weight = Weight.conv_weight('conv13_weight', 512, 512, 3, 3)
        self.conv13_bias = Weight.bias('conv13_bias', 512, 'zeros')

        self.full1_weight = Weight.fully_weight('full1_weight', full1_input_size, 4096)
        self.full1_bias = Weight.bias('full1_bias', 4096, 'truncated_normal')

        self.full2_weight = Weight.fully_weight('full2_weight', 4096, 4096)
        self.full2_bias = Weight.bias('full2_bias', 4096, 'truncated_normal')

        self.full3_weight = Weight.fully_weight('full3_weight', 4096, self.target_num)
        self.full3_bias = Weight.bias('full3_bias', self.target_num, 'truncated_normal')
    
    elif self.load_weights == 'yes':
      weights = np.load('/content/drive/My Drive/saved_model/vgg16_weights.npz')
      keys = sorted(weights.keys())

      with tf.name_scope('{0}'.format(self.DNN_name)):
        print('Train with pre-trained model')
        self.conv1_weight = tf.Variable(weights[keys[0]], name='conv1_weight')
        self.conv1_bias = tf.Variable(weights[keys[1]], name='conv1_bias')

        self.conv2_weight = tf.Variable(weights[keys[2]], name='conv2_weight')
        self.conv2_bias = tf.Variable(weights[keys[3]], name='conv2_bias')

        self.conv3_weight = tf.Variable(weights[keys[4]], name='conv3_weight')
        self.conv3_bias = tf.Variable(weights[keys[5]], name='conv3_bias')

        self.conv4_weight = tf.Variable(weights[keys[6]], name='conv4_weight')
        self.conv4_bias = tf.Variable(weights[keys[7]], name='conv4_bias')

        self.conv5_weight =  tf.Variable(weights[keys[8]], name='conv5_weight')
        self.conv5_bias =  tf.Variable(weights[keys[9]], name='conv5_bias')

        self.conv6_weight = tf.Variable(weights[keys[10]], name='conv6_weight')
        self.conv6_bias = tf.Variable(weights[keys[11]], name='conv6_bias')

        self.conv7_weight = tf.Variable(weights[keys[12]], name='conv7_weight')
        self.conv7_bias = tf.Variable(weights[keys[13]], name='conv7_bias')

        self.conv8_weight = tf.Variable(weights[keys[14]], name='conv8_weight')
        self.conv8_bias = tf.Variable(weights[keys[15]], name='conv8_bias')

        self.conv9_weight = tf.Variable(weights[keys[16]], name='conv9_weight')
        self.conv9_bias = tf.Variable(weights[keys[17]], name='conv9_bias')

        self.conv10_weight = tf.Variable(weights[keys[18]], name='conv10_weight')
        self.conv10_bias = tf.Variable(weights[keys[19]], name='conv10_bias')

        self.conv11_weight = tf.Variable(weights[keys[20]], name='conv11_weight')
        self.conv11_bias = tf.Variable(weights[keys[21]], name='conv11_bias')

        self.conv12_weight = tf.Variable(weights[keys[22]], name='conv12_weight')
        self.conv12_bias = tf.Variable(weights[keys[23]], name='conv12_bias')

        self.conv13_weight = tf.Variable(weights[keys[24]], name='conv13_weight')
        self.conv13_bias = tf.Variable(weights[keys[25]], name='conv13_bias')
        
        self.full1_weight = Weight.fully_weight('full1_weight', full1_input_size, 4096)
        self.full1_bias = Weight.bias('full1_bias', 4096, 'truncated_normal')

        self.full2_weight = tf.Variable(weights[keys[28]], name='full1_weight')
        self.full2_bias = tf.Variable(weights[keys[29]], name='conv13_bias')
        
        
        self.full3_weight = Weight.fully_weight('full3_weight', 4096, self.target_num)
        self.full3_bias = Weight.bias('full3_bias', self.target_num, 'truncated_normal')

      
  # 我們必須在tf.function中specifiy input data的shpape(這裡都是None 不受限制)，否則tf.function會把input data的shape定死(也就是在training的時候吃到的shape)，到時再進行testing的時候會出現shape不合的問題(因為testing的時候並不一定會用training時的batch size來做為輸入)
  @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
  def __call__(self, input_data):  
    self.input_data = input_data

    with tf.name_scope('{0}'.format(self.DNN_name)):
      tf.keras.Input(shape=self.input_data_shape)
      conv1 = tf.nn.conv2d(self.input_data, self.conv1_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv1')
      relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_bias))
      

      
      conv2 = tf.nn.conv2d(relu1, self.conv2_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv2')
      relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_bias))

      max_pool1 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')

      
      conv3 = tf.nn.conv2d(max_pool1, self.conv3_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv3')
      relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_bias))

      
      conv4 = tf.nn.conv2d(relu3, self.conv4_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv4')
      relu4 = tf.nn.relu(tf.nn.bias_add(conv4, self.conv4_bias))
      
      max_pool2 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')
      
      
      conv5 = tf.nn.conv2d(max_pool2, self.conv5_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv5')
      relu5 = tf.nn.relu(tf.nn.bias_add(conv5, self.conv5_bias))

      
      conv6 = tf.nn.conv2d(relu5, self.conv6_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv6')
      relu6 = tf.nn.relu(tf.nn.bias_add(conv6, self.conv6_bias))

      
      conv7 = tf.nn.conv2d(relu6, self.conv7_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv7')
      relu7 = tf.nn.relu(tf.nn.bias_add(conv7, self.conv7_bias))
      
      max_pool3 = tf.nn.max_pool(relu7, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')
      
      
      conv8 = tf.nn.conv2d(max_pool3, self.conv8_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv8')
      relu8 = tf.nn.relu(tf.nn.bias_add(conv8, self.conv8_bias))

      
      conv9 = tf.nn.conv2d(relu8, self.conv9_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv9')
      relu9 = tf.nn.relu(tf.nn.bias_add(conv9, self.conv9_bias))

      
      conv10 = tf.nn.conv2d(relu9, self.conv10_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv10')
      relu10 = tf.nn.relu(tf.nn.bias_add(conv10, self.conv10_bias))

      max_pool4 = tf.nn.max_pool(relu10, ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1], padding='SAME')
      
      
      conv11 = tf.nn.conv2d(max_pool4, self.conv11_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv11')
      relu11 = tf.nn.relu(tf.nn.bias_add(conv11, self.conv11_bias))

      
      conv12 = tf.nn.conv2d(relu11, self.conv12_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv12')
      relu12 = tf.nn.relu(tf.nn.bias_add(conv12, self.conv12_bias))

      
      conv13 = tf.nn.conv2d(relu12, self.conv13_weight, strides=[1, 1, 1, 1], padding='SAME', name='conv13')
      relu13 = tf.nn.relu(tf.nn.bias_add(conv13, self.conv13_bias))
      
      if self.droupout == 'yes':
          # flatten
          final_conv_shape = tf.shape(relu13)
          final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
          flat_output = tf.reshape(relu13, [final_conv_shape[0], final_shape])
          
          
          full1 = tf.nn.relu(tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias), name = 'full1')
          relu14 = tf.nn.relu(full1)
          relu14 = tf.nn.dropout(tf.nn.relu(relu14), self.dropout_rate)
          
          
          full2 = tf.add(tf.matmul(relu14, self.full2_weight), self.full2_bias, name = 'full2')
          relu15 = tf.nn.relu(full2)
          relu15 = tf.nn.dropout(tf.nn.relu(relu15), self.dropout_rate)
          
          
          full3 = tf.add(tf.matmul(relu15, self.full3_weight), self.full3_bias, name = 'full3')
     
      
      if self.droupout == 'no':
          # flatten
          final_conv_shape = tf.shape(relu13)
          final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
          flat_output = tf.reshape(relu13, [final_conv_shape[0], final_shape])
          
          
          full1 = tf.nn.relu(tf.add(tf.matmul(flat_output, self.full1_weight), self.full1_bias), name = 'full1')
          relu14 = tf.nn.relu(full1)
          
          
          full2 = tf.add(tf.matmul(relu14, self.full2_weight), self.full2_bias, name = 'full2')
          relu15 = tf.nn.relu(full2)
          
          
          full3 = tf.add(tf.matmul(relu15, self.full3_weight), self.full3_bias, name = 'full3')
          
      
      return(full3)

  def train_var(self):
    if self.load_weights == 'no':
      train_var = [self.conv1_weight, self.conv2_weight, self.conv3_weight, self.conv4_weight, self.conv5_weight, self.conv6_weight, self.conv7_weight, self.conv8_weight, self.conv9_weight, self.conv10_weight, self.conv11_weight, self.conv12_weight, self.conv13_weight, self.conv1_bias, self.conv2_bias, self.conv3_bias, self.conv4_bias, self.conv5_bias, self.conv6_bias, self.conv7_bias, self.conv8_bias, self.conv9_bias, self.conv10_bias, self.conv11_bias, self.conv12_bias, self.conv13_bias, self.full1_weight, self.full2_weight, self.full3_weight, self.full1_bias, self.full2_bias, self.full3_bias]
    elif self.load_weights == 'yes':
      train_var = [self.conv11_weight, self.conv12_weight, self.conv13_weight, self.conv11_bias, self.conv12_bias, self.conv13_bias, self.full1_weight, self.full2_weight, self.full3_weight, self.full1_bias, self.full2_bias, self.full3_bias]
    return(train_var)