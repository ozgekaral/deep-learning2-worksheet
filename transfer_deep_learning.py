# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 14:54:13 2023

@author: user202
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob

train_path='transform deep learning/Training/'
test_path='transform deep learning/Test/'

img=load_img(train_path + 'Banana/0_100.jpg')
data=img_to_array(image)
print(data.shape)
all_classes=glob(train_path+'/*')
numberofclass=len(glob(train_path+'/*'))

#Model 
model=VGG16()
layers_list=model.layers

for i in range(len(layers_list)-1):
    Sequential().add(layers_list[i])

for j in Sequential().layers:
    j.trainable = False

Sequential().add(Dense(numberofclass, activation='softmax'))
Sequential().compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[accuracyracy])







