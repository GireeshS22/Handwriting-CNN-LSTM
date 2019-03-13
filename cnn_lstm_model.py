# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:23:55 2019

@author: Gireesh Sundaram
"""

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Activation, Reshape
from keras.layers import Bidirectional, LSTM
from keras.layers.normalization import BatchNormalization

from configuration import window_height, window_width, MPoolLayers_ALL, LastFilters, NUnits

import tensorflow as tf

import math

#%%
FV = int(window_height / math.pow(2, MPoolLayers_ALL))
NFeatures = FV * LastFilters

#%%
model = Sequential()
model.add(Convolution2D(input_shape=(window_height, window_width, 1), filters=64, kernel_size=(1,1)))   #Convolution 1
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=64, kernel_size=(1,1))) #Convolution 2
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,2)))

model.add(Convolution2D(filters=128, kernel_size=(1,1)))    #Convolution 3
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=128, kernel_size=(1,1)))    #Convolution 4
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,2)))

model.add(Convolution2D(filters=256, kernel_size=(1,1)))    #Convolution 5
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=256, kernel_size=(1,1)))    #Convolution 6
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=256, kernel_size=(1,1)))    #Convolution 7
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 8
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 9
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 10
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 11
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 12
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(Convolution2D(filters=512, kernel_size=(1,1)))    #Convolution 13
model.add(BatchNormalization(axis=-1))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

model.add(Reshape((LastFilters * FV, 16)))

model.add(Bidirectional(LSTM(units = NUnits, return_sequences=True)))   #Bi-LSTM 1
model.add(Bidirectional(LSTM(units = NUnits, return_sequences=True)))   #Bi-LSTM 1
model.add(Bidirectional(LSTM(units = NUnits)))  #Bi-LSTM 3

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.summary()