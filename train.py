# import the self-define module
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('{0}/lib'.format(current_path))
import tensorflow as tf
import numpy as np
import vgg
import ops
import h5py
import timeit
import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--save_name", type=str,default=None, dest='save_name', help="The name of the saved model and training result")
parser.add_argument("--iteration", type=int,default= None, dest='iteration', help="Setting the iteration number of the training")
parser.add_argument("--data", type=str,default='fer2013', dest='data', help="Choosing the training data, default for 48x48 data, type resized for 112x112 data.")
parser.add_argument("--drop_rate", type=float,default=0, dest='drop_rate', help="Define the drop rate. Default is 0 for not enable dropout.")
args = parser.parse_args()
drop_rate = tf.constant(args.drop_rate, dtype=tf.float32)


if args.data == 'fer2013':
  fh = h5py.File('{0}/Training_data/fer2013.h5'.format(current_path), 'r')
elif args.data == 'resized':
  fh = h5py.File('{0}/Training_data/fer2013_112_112.h5'.format(current_path), 'r')
else:
  raise ValueError('argument --data: Please type resized or just leave this argument empty')
  
# emotion_label = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}
trainX = np.array(fh.get('Train/trainX'))
trainY = np.array(fh.get('Train/trainY'))
trainValX = np.array(fh.get('TrainVal/trainValX'))
trainValY = np.array(fh.get('TrainVal/trainValY'))
fh.close()


print('train:\t',trainX.shape,trainY.shape)
print('trainVal:',trainValX.shape,trainValY.shape)

if len(np.shape(trainX)) < 4:
  trainX = np.expand_dims(trainX, axis=3)
  trainValX = np.expand_dims(trainValX, axis=3)
  print('Expand the channel dimension for TensorFlow')

model = vgg.VGG16(7, drop_rate, 'Emotion_Recognition')
optimizer = tf.optimizers.Adam(learning_rate=0.001)

save_name = args.save_name
model_save_path = '{0}/saved_model/{1}'.format(current_path, save_name)
logdir = os.path.join('{0}/tensorboard/{1}'.format(current_path, save_name), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
log_writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=False)


iteration = args.iteration
one_hot = 'no'
batch_size = 128

train_loss = []
train_acc = []
valid_acc = []
valid_loss = []
ini_train_loss = 1000000
ini_valid_loss = 1000000
s = timeit.default_timer()

for i in tf.range(iteration):
  rand_index = np.random.choice(len(trainX), size=batch_size, replace=False)
  rand_x = tf.convert_to_tensor(trainX[rand_index], dtype=tf.float32)
  rand_y = tf.convert_to_tensor(trainY[rand_index], dtype=tf.int32) # label
  temp_train_loss = ops.train(model, rand_x, rand_y, optimizer)
  
  if (i+1) % 20 == 0:
    # Record and print results
    batch_predictions = ops.predict(model, rand_x)
    valid_index = np.random.choice(len(trainValY), size=batch_size)
    valid_x = tf.convert_to_tensor(trainValX[valid_index], dtype=tf.float32)
    valid_y = tf.convert_to_tensor(trainValY[valid_index], dtype=tf.int32)
    valid_predictions = ops.predict(model, valid_x)
  
    batch_predictions = tf.argmax(batch_predictions, axis=1, output_type=tf.int32)
    temp_train_acc = tf.reduce_mean(tf.cast(tf.equal(batch_predictions, rand_y), tf.float32))
  
    valid_predictions = tf.argmax(valid_predictions, axis=1, output_type=tf.int32)
    temp_valid_acc = tf.reduce_mean(tf.cast(tf.equal(valid_predictions, valid_y), tf.float32))
      
    temp_valid_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ops.predict(model, valid_x), labels=valid_y))
  
    if temp_train_loss < ini_train_loss:
      ini_train_loss = temp_train_loss
      if temp_valid_loss < ini_valid_loss:
        model.save(model_save_path)
        ini_valid_loss = temp_valid_loss
        saved_iteration = i
  
    train_loss.append(temp_train_loss)
    train_acc.append(temp_train_acc)
    valid_acc.append(temp_valid_acc)
    valid_loss.append(temp_valid_loss)
    acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_valid_acc]
    acc_and_loss = [np.round(x,2) for x in acc_and_loss]
    print('Iteration # {}. Train Loss: {:.2f}. Train Acc (Val Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

    with log_writer.as_default():
      tf.summary.scalar('train_loss', temp_train_loss, step=tf.cast(i, dtype=tf.int64))
      tf.summary.scalar('valid_loss', temp_valid_loss, step=tf.cast(i, dtype=tf.int64))
      tf.summary.scalar('train_acc', temp_train_acc, step=tf.cast(i, dtype=tf.int64))
      tf.summary.scalar('valid_acc', temp_valid_acc, step=tf.cast(i, dtype=tf.int64))

e = timeit.default_timer()
print(e - s)

training_result_path = '{0}/Training_result/{1}.h5'.format(current_path, save_name)
fh = h5py.File(training_result_path, 'w')

fh.create_dataset('/Train/loss', data=train_loss)
fh.create_dataset('/Train/acc', data=train_acc)
fh.create_dataset('/Train/iteration', data=iteration)
fh.create_dataset('/Train/saved_iteration', data=saved_iteration.numpy())
fh.create_dataset('/Validation/loss', data=valid_loss)
fh.create_dataset('/Validation/acc', data=valid_acc)

fh.close()