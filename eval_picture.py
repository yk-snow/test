#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:20:49 2020

@author: kimurayasuhisa
"""

#綾鷹を選ばせるプログラム

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

model = model_from_json(open('model_data/test_predict_ext150.json').read())
model.load_weights('model_data/test_predict_ext150.hdf5')
print('model loaded')
#%%

categories = ["PS4","ぬいぐるみ","イス"]
#%%
#画像を読み込む
print("input : ")
img_path = str(input())

img = image.load_img(img_path,target_size=(250, 250, 3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#予測
features = model.predict(x)

img.show()

#予測結果によって処理を分ける
if features[0,0] == 1:
    print ("選ばれたのはPS4でした")

elif features[0,1] == 1:
    print ("選ばれたのは、ぬいぐるみでした")
    
elif features[0,2] == 1:
    print ("選ばれたのは、イスでした")

else:
    print("Error")