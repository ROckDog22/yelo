# !/usr/bin/env Python3
# -*- coding: utf-8 -*-
# @Author   : 汪逢生
# @FILE     : re.py
# @Time     : 2020/7/26 19:43
# @Software : PyCharm

import base64
import requests
img_file = open('japanese.jpg','rb')
img_b64encode = base64.b64encode(img_file.read())  # 使用base64進行加密
data = {
    'img':img_b64encode
}
url = 'http://127.0.0.1:5000/img/send_img'
res = requests.post(url=url,data=data)
print(res.content)

