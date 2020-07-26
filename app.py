# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : app.py.py
# @Time     : 2020/7/26 15:58
# @Software : PyCharm

from flask import Flask
import json
import requests
from keras.preprocessing import image
from  flask import request
import io
from PIL import Image
import base64
import time
app = Flask(__name__)
app.debug=True

@app.route('/img/send_img', methods=['POST', "GET"])
def send_img():
    if request.method == 'POST':
        img = request.form.get('img')
        img_b64decode = base64.b64decode(img)
        img = io.BytesIO(img_b64decode)
        img = Image.open(img)
        path = './static/img/'+str(time.time())+'.jpg'
        img.save(path)
        img_new = image.img_to_array(image.load_img(path, target_size=(224, 224))) / 255.
        img_new = img_new.astype('float16')
        param = {"instances": [{'input_1': img_new.tolist()}]}
        param = json.dumps(param)
        res = requests.post('http://localhost:8501/v1/models/resnet:predict', data=param)
        # res = requests.post('http://yellow.laoding.online:8001/v1/models/resnet:predict', data=param)
        return res.content
    else:
        return "error"



@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
