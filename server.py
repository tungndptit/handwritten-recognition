# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import numpy as np
from keras.models import model_from_json
from skimage.transform import resize
from skimage.io import imread
import config

app = Flask(__name__)
CORS(app)

images_folder = "images"
classes = [chr(c) for c in range(ord('a'), ord('z') + 1)]

image_index = 1
input_size = 32


def load_model():
    f = open(config.TRAIN.model_file)
    json_content = f.read()
    f.close()
    cnn_model = model_from_json(json_content)
    cnn_model.load_weights(config.TRAIN.model_weight_file)
    return cnn_model


model = load_model()


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/images/recognize', methods=['POST'])
def recognize():
    img_path = "web_images.png"
    image_data = request.json['image_data']
    image_data = image_data[22:]                   # remove header padding
    f = open(img_path, 'wb')
    f.write(base64.b64decode(image_data))
    f.close()

    img = imread(img_path)
    img = img[:, :, -1]
    img = resize(img, (input_size, input_size), preserve_range=True)
    img_x = img.astype('float32') / 255.0

    # img_x = img_x.reshape((1, input_size * input_size)) # using for MLP Model

    y = model.predict(img_x)[0]
    print('Input {} , output : {}'.format(img_path, classes[np.argmax(y)]))

    os.remove(img_path)
    return jsonify(classes[np.argmax(y)])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, threaded=False)