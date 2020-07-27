# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : tansfer.py
# @Time     : 2020/7/27 15:31
# @Software : PyCharm

import tensorflow as tf

# 首先使用tf.keras的load_model来导入模型h5文件
model_path = 'nsfw_mobilenet2.224x224.h5'
model = tf.keras.models.load_model(model_path)
model.save('models/resnet/', save_format='tf')  # 导x出tf格式的模型文件