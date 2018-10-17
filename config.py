# -*- coding: utf-8 -*-

import os


class RESOURCE(object):
    dir = 'resource'


class IMAGE(object):
    dir = os.path.join(RESOURCE.dir, 'images')
    size_w = 32
    size_h = 32
    size = (size_w, size_h)


class TRAIN(object):
    image_np_dir = os.path.join(RESOURCE.dir, 'images_np')
    x_test_np_file = os.path.join(image_np_dir, 'x_test.np')
    y_test_np_file = os.path.join(image_np_dir, 'y_test.np')
    x_train_np_file = os.path.join(image_np_dir, 'x_train.np')
    y_train_np_file = os.path.join(image_np_dir, 'y_train.np')
    model_dir = os.path.join(RESOURCE.dir, 'models')
    model_file = os.path.join(model_dir, 'model_cnn.json')
    model_weight_file = os.path.join(model_dir, 'model_cnn_weight.h5')
    data_split_rate = 0.7


class TEST(object):
    dir = os.path.join(RESOURCE.dir, 'test_images')


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


create_dir_if_not_exists(RESOURCE.dir)
create_dir_if_not_exists(TRAIN.image_np_dir)
create_dir_if_not_exists(TRAIN.model_dir)
