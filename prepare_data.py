# -*- coding: utf-8 -*-

import os
import numpy as np
from random import random, randint, shuffle
from skimage.transform import rotate, resize
from skimage.io import imread
from keras.utils import to_categorical
import config

num_img_per_class = 2000
train_factor = 0.7

min_scale = 0.75
max_scale = 1.35


def gen_img(img_path):
    image = imread(img_path)
    image = image[:, :, -1]
    rotation = 20 * random() - 10
    image = rotate(image, rotation, preserve_range=True)
    img_size = len(image)
    h_scan = np.max(image, axis=0)
    x1 = np.min(np.where(h_scan > 0))
    x2 = np.max(np.where(h_scan > 0))

    v_scan = np.max(image, axis=1)
    y1 = np.min(np.where(v_scan > 0))
    y2 = np.max(np.where(v_scan > 0))

    dx, dy = x2 - x1, y2 - y1
    min_size = max(dx, dy, int(img_size / max_scale))
    max_size = int(img_size / min_scale)
    new_size = randint(min_size, max_size)
    offset_x = randint(0, new_size - dx)
    offset_y = randint(0, new_size - dy)
    new_img = np.zeros((new_size, new_size), dtype=np.uint8)
    new_img[offset_y:offset_y + dy, offset_x:offset_x + dx] = image[y1:y2, x1:x2]
    return new_img


classes = os.listdir(config.IMAGE.dir)
num_classes = len(classes)
print('Number class', num_classes)

map_img_paths = {}
for class_name in classes:
    sub_folder = os.path.join(config.IMAGE.dir, class_name)
    img_files = [f for f in os.listdir(sub_folder)
                 if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    map_img_paths[class_name] = [os.path.join(sub_folder, f) for f in img_files]

data = []

for i in range(len(classes)):
    class_name = classes[i]
    img_list = map_img_paths[class_name]

    print('Generate images for class ', class_name)

    for j in range(num_img_per_class):
        img = gen_img(img_list[j % len(img_list)])
        img = resize(img, config.IMAGE.size, preserve_range=True)
        img = img.astype('uint8')
        data.append((img, i))

# split data for training and testing
shuffle(data)
n_train = int(len(data) * config.TRAIN.data_split_rate)

train_data = data[:n_train]
test_data = data[n_train:]

x_train = np.array([item[0] for item in train_data], dtype=np.uint8)
y_train = np.array([to_categorical(item[1], num_classes) for item in train_data], dtype=np.uint8)

x_test = np.array([item[0] for item in test_data], dtype=np.uint8)
y_test = np.array([to_categorical(item[1], num_classes) for item in test_data], dtype=np.uint8)

x_train.tofile(config.TRAIN.x_train_np_file)
y_train.tofile(config.TRAIN.y_train_np_file)
x_test.tofile(config.TRAIN.x_test_np_file)
y_test.tofile(config.TRAIN.y_test_np_file)

