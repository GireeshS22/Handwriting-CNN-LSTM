# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:23:55 2019

@author: Gireesh Sundaram
"""

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization

from configuration import window_height, window_width

import tensorflow as tf

#%%
model = Sequential()
model.add(Convolution2D(input_shape=(window_height, window_width, 1), filters=64, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=64, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,2)))

model.add(Convolution2D(filters=128, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=128, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,2)))

model.add(Convolution2D(filters=256, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=256, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=256, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))



model.compile(optimizer="adam", loss="categorical_crossentropy")
model.summary()