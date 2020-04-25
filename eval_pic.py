#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:20:35 2020

@author: wanpyon
"""
# モデルのロード
from keras.models import model_from_json

model = model_from_json(open('model_data/test_predict.json').read())
model.load_weights('model_data/test_predict.hdf5')

print("--- Complete loding !! ---")

#選ばせるプログラム

from keras import models
from keras.preprocessing import image
import numpy as np

categories = ["PS4","ぬいぐるみ", "イス"]
#画像を読み込む
print("input :")
img_path = str(input())
print("--- input checked!! ---")
img = image.load_img(img_path,target_size=(150, 150, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

#予測結果によって処理を分ける
if features[0,0] == 1:
    print ("選ばれたのは、PS4でした。")

elif features[0,1] == 1:
    print ("選ばれたのは、ぬいぐるみでした。")

elif features[0,2] == 1:
    print ("選ばれたのは、イスでした。")

else :
    print ("Error。")

