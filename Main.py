from keras import regularizers
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential

base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(227,227,3))

model = Sequential()
model.add(base)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001)))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])