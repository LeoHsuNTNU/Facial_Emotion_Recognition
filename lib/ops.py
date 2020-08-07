import tensorflow as tf
import numpy as np

class operator():
  def __init__(self):
    pass
  
  def accuracy(self, logits, labels):
    self.logits = logits
    self.labels = labels
    preds = tf.argmax(self.logits, axis=1) 
    labels = tf.argmax(self.labels, axis=1)
    return(tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32)))

  def accuracy_sparse(self, logits, labels):
    self.logits = logits
    self.labels = labels
    preds = tf.argmax(logits, axis=1)
    return(tf.reduce_mean(tf.cast(tf.equal(preds, self.labels), tf.float32)))

  @tf.function
  def train(self, model, inputX, inputY, optimizer, dropout, dropout_rate):
    self.model = model
    self.inputX = inputX
    self.inputY = inputY
    self.optimizer = optimizer
    self.dropout = dropout
    self.dropout_rate = dropout_rate
    
    with tf.GradientTape() as tape:
        temp_train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model(self.inputX, dropout, dropout_rate), labels=self.inputY))
    trainable_variables = self.model.train_var()
    grads = tape.gradient(temp_train_loss, trainable_variables) 
    
    optimizer.apply_gradients(zip(grads, trainable_variables))
    
    return(temp_train_loss)
  
  @tf.function
  def predict(self, model, inputX):
    self.model = model
    self.inputX = inputX
    
    predictions = tf.nn.softmax(self.model(self.inputX, False))

    return(predictions)
