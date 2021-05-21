
from keras import regularizers
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
import random
import os, shutil

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# 폴더가 없으면 폴더를 만듬
dataset_dir = './dataset'
dataset_positive = os.path.join(dataset_dir, 'positive')
dataset_negative = os.path.join(dataset_dir, 'negative')

dest_dir = './dataset_sorted'
Path(dest_dir).mkdir(parents=True, exist_ok=True)

train_dir = os.path.join(dest_dir, 'train')
validation_dir = os.path.join(dest_dir, 'validation')
test_dir = os.path.join(dest_dir, 'test')
Path(train_dir).mkdir(parents=True, exist_ok=True)
Path(validation_dir).mkdir(parents=True, exist_ok=True)
Path(test_dir).mkdir(parents=True, exist_ok=True)

train_positive_dir = os.path.join(train_dir, 'positive')
train_negative_dir = os.path.join(train_dir, 'negative')
Path(train_positive_dir).mkdir(parents=True, exist_ok=True)
Path(train_negative_dir).mkdir(parents=True, exist_ok=True)

validation_positive_dir = os.path.join(validation_dir, 'positive')
validation_negative_dir = os.path.join(validation_dir, 'negative')
Path(validation_positive_dir).mkdir(parents=True, exist_ok=True)
Path(validation_negative_dir).mkdir(parents=True, exist_ok=True)

test_positive_dir = os.path.join(test_dir, 'positive')
test_negative_dir = os.path.join(test_dir, 'negative')
Path(test_positive_dir).mkdir(parents=True, exist_ok=True)
Path(test_negative_dir).mkdir(parents=True, exist_ok=True)

fnames = os.listdir(dataset_positive)
random.shuffle(fnames)

if len(os.listdir(validation_positive_dir)) < 1000:
    for i, f in enumerate(fnames):
        if i < 1000:
            src = os.path.join(dataset_positive, f)
            dst = os.path.join(validation_positive_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)
        elif i<2000:
            src = os.path.join(dataset_positive, f)
            dst = os.path.join(test_positive_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)
        else:
            src = os.path.join(dataset_positive, f)
            dst = os.path.join(train_positive_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)

fnames = os.listdir(dataset_negative)
random.shuffle(fnames)

if len(os.listdir(validation_negative_dir)) < 1000:
    for i, f in enumerate(fnames):
        if i < 1000:
            src = os.path.join(dataset_negative, f)
            dst = os.path.join(validation_negative_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)
        elif i<2000:
            src = os.path.join(dataset_negative, f)
            dst = os.path.join(test_positive_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)
        else:
            src = os.path.join(dataset_negative, f)
            dst = os.path.join(train_negative_dir, str(i) + ".jpg")
            shutil.copyfile(src, dst)


train_datagen = ImageDataGenerator()
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(256,256))
validation_generator = validation_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=(256,256))

net_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

net_base.trainable = True
set_trainable = False
flag = False
if(flag):
    for layer in net_base.layers:
        if layer.name== "block8_10_conv":
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

model = Sequential()
model.add(net_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator,
                              validation_steps=50)

history_dict = history.history
accuracy = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epoch = range(1, len(accuracy) + 1)

plt.plot(epoch, accuracy, 'bo', label='Training Accuracy')
plt.plot(epoch, val_acc, 'b', label='Validation Accuracy')
plt.title('Training, Validation Accuracy')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.show()
