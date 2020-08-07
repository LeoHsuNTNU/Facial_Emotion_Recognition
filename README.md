# Facial_Emotion_Recognition
Classifying the fer2013 facial emotion dataset with tensorflow.

## Training
Run
```
python ./train.py --save_name <string> --data <string> --iteration <int> --enable_dropout <bool> --drop_rate <float>
```
for tain the VGG16 network.

If you want train the data with resized 112x112 dataset (resized via OpenCV), choose ```--data=resized```, if not, just leave it empty. 

If the arrgument ```--enable_dropout=False```, the ```--drop_rate``` arrgument will not work.

The trained model wii be saved in ```./saved_model```, and the training process will be saved in ```./Training_result```.

## Testing
Run
```
python ./test.py --save_name <string> --data <string> --testing_batch <int> --save_test_result <bool>
```
for testing the trained data.

Choose ```--save_test_result``` for save the testing result or not. If true, it will be saved in the same file which in ```./Training_result```
