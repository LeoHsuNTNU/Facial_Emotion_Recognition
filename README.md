# Facial_Emotion_Recognition
Classifying the fer2013 facial emotion dataset with tensorflow.

## Training
run
```
python ./train.py --save_name <string> --data <string> --iteration <int> --enable_dropout <bool> --drop_rate <float>
```
for tain the VGG16 network. If you want train the data with resized 112x112 dataset (resized via OpenCV), choose ```--data=resized```. If the arrgument ```--enable_dropout=False```, the ```--drop_rate``` arrgument will not work.
The trained model wii be saved in ./saved_model, and the training process will be saved in Training_result.

## Testing
run
```
