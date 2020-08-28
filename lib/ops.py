# -*- coding: utf-8 -*-
"""ops

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L966fMwAcDWRyEmxumAWhmzmMryoDvhq
"""

import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('{0}/lib'.format(current_path))
import tensorflow as tf
import numpy as np
import network

  
def accuracy(logits, labels):
  logits = logits
  labels = labels
  preds = tf.argmax(logits, axis=1) 
  labels = tf.argmax(labels, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

def accuracy_sparse(logits, labels):
  logits = logits
  labels = labels
  preds = tf.argmax(logits, axis=1)
  return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

@tf.function(experimental_relax_shapes=True)
def train(model, inputX, inputY, optimizer):
  with tf.GradientTape() as tape:
      temp_train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model(inputX, tf.cast(True, tf.bool)), labels=inputY))
  trainable_variables = model.trainable_weights
  grads = tape.gradient(temp_train_loss, trainable_variables) 
  
  optimizer.apply_gradients(zip(grads, trainable_variables))
  
  return temp_train_loss

@tf.function(experimental_relax_shapes=True)
def predict(model, inputX):
  model = model
  inputX = inputX
  
  predictions = tf.nn.softmax(model(inputX, False))

  return predictions