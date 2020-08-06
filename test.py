# import the self-define module
import os
current_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append('{0}/lib'.format(current_path))
import tensorflow as tf
import h5py
import timeit
import numpy as np
import ops
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_name", type=str,default=None, dest='save_name', help="The name of the saved model and training result")
parser.add_argument("--testing_batch", type=int,default= None, dest='testing_batch', help="Setting batch number for testing, avoiding the OOM error")
parser.add_argument("--data", type=str,default='fer2013', dest='data', help="Choosing the training data, default for 48x48 data, type resized for 112x112 data.")
parser.add_argument("--save_test_result", type=str,default=None, dest='save_test_result', help="Save the test result or not, please type yes for saveing or leave empty for not saving")
args = parser.parse_args()

if args.data == 'fer2013':
  fh = h5py.File('{0}/Training_data/fer2013.h5'.format(current_path), 'r')
elif args.data == 'resized':
  fh = h5py.File('{0}/Training_data/fer2013_112_112.h5'.format(current_path), 'r')
else:
  raise valueerror('argument --data: Please type resized or just leave this argument empty')

testX = np.array(fh.get('Test/testX'))
testY = np.array(fh.get('Test/testY'))

fh.close()

save_name = args.save_name
model_save_path = '{0}/saved_model/{1}'.format(current_path, save_name)
reload_model = tf.saved_model.load(model_save_path)

test_batch = args.testing_batch
splited_num = int(np.shape(testX)[0] / test_batch)
start_index = 0
test_acc = []
one_hot = 'no'
s = timeit.default_timer()
for i in tf.range(splited_num):
  test_index = np.arange(start_index, start_index + test_batch)
  batch_testx = tf.convert_to_tensor(testX[test_index], dtype=tf.float32)
  batch_testy = tf.convert_to_tensor(testY[test_index], dtype=tf.int32)
  test_predictions = reload_model(batch_testx)
  
  if one_hot == 'no':
    test_predictions = tf.argmax(test_predictions, axis=1, output_type=tf.int32)
    temp_test_acc = tf.reduce_mean(tf.cast(tf.equal(test_predictions, batch_testy), tf.float32))
  
  elif one_hot == 'yes':
    test_predictions  = tf.argmax(test_predictions, axis=1, output_type=tf.int32) 
    labels = tf.argmax(batch_testy, axis=1, output_type=tf.int32)
    temp_test_acc = tf.reduce_mean(tf.cast(tf.equal(test_predictions, labels), tf.float32))
  
  test_acc.append(temp_test_acc)
  start_index = start_index + test_batch
  #print(temp_test_acc)
test_acc_real = tf.make_ndarray(tf.make_tensor_proto(sum(test_acc) / splited_num))
tf.print('Test Acc: ', test_acc_real * 100, '%')


e = timeit.default_timer()
print('Execution Time: ', e - s, '(s)')


if args.save_test_result == 'yes':
  result_path = '{0}/Training_result/{1}.h5'.format(current_path, save_name)
  fh = h5py.File(result_path, 'a')
  
  fh.create_dataset('/Test/acc', data=test_acc_real)
  fh.create_dataset('/Test/execution_Time', data=e - s)
  
  fh.close()
