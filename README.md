# Facial_Emotion_Recognition
Classifying the fer2013 facial emotion dataset with VGG16, which constructed with TensorFlow low level api tf.nn.

# Update 2020/08/28
I change the coding style into custom keras model and custom keras layers, rewriting ./lib/vgg.py and ./lib/ops.py. The custom layers are defined in ./lib/network.py

I added the batch normalization layers in VGG16, the test accuracy increased from 54.47% to 61.55%
## Training
Run
```
python ./train.py --save_name <string> --iteration <int> --data <string> --drop_rate <float>
```
for taining the VGG16 network.

If you want to train the network with resized 112x112 dataset (resized via OpenCV), choose ```--data=resized```, if not, just leave it empty. 

The trained model wii be saved in ```./saved_model```, and the training process will be saved in ```./Training_result```.

## Testing
Run
```
python ./test.py --save_name <string> --data <string> --testing_batch <int> --save_test_result <bool>
```
for testing the trained model.

Choose ```--save_test_result``` for save the testing result or not. If true, it will be saved in the same file which in ```./Training_result```
