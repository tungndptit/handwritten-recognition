# -*- coding: utf-8 -*-
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import config

classes = [folder for folder in os.listdir(config.IMAGE.dir)]
num_classes = len(classes)

x_train = np.fromfile(config.TRAIN.x_train_np_file, dtype=np.uint8).reshape(-1, config.IMAGE.size_w,
                                                                            config.IMAGE.size_h, 1)
y_train = np.fromfile(config.TRAIN.y_train_np_file, dtype=np.uint8).reshape(-1, num_classes)
x_test = np.fromfile(config.TRAIN.x_test_np_file, dtype=np.uint8).reshape(-1, config.IMAGE.size_w, config.IMAGE.size_h,
                                                                          1)
y_test = np.fromfile(config.TRAIN.y_test_np_file, dtype=np.uint8).reshape(-1, num_classes)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(config.IMAGE.size_w, config.IMAGE.size_h, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=True)

f = open(config.TRAIN.model_file, 'w')
f.write(model.to_json())
f.close()

model.save_weights(config.TRAIN.model_weight_file)
print('Done!')
