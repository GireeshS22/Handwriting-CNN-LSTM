# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:23:55 2019

@author: Gireesh Sundaram
"""

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Reshape
from keras.layers import Bidirectional, LSTM
from keras.layers.normalization import BatchNormalization

from configuration import window_height, window_width, MPoolLayers_ALL, LastFilters, NUnits

import tensorflow as tf

import math

#%%
FV = int(window_height / math.pow(2, MPoolLayers_ALL))
NFeatures = FV * LastFilters

#%%
input_data = Input(shape=(window_height, window_width, 1))

convolution1 = Conv2D(filters=64, kernel_size=(1,1))(input_data)
convolution1 = BatchNormalization(axis = -1)(convolution1)
convolution1 = Activation("relu")(convolution1)

convolution2 = Conv2D(filters=64, kernel_size=(1,1))(convolution1)
convolution2 = BatchNormalization(axis = -1)(convolution2)
convolution2 = Activation("relu")(convolution2)

pooling1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(convolution2)

convolution3 = Conv2D(filters=128, kernel_size=(1,1))(pooling1)
convolution3 = BatchNormalization(axis = -1)(convolution3)
convolution3 = Activation("relu")(convolution3)

convolution4 = Conv2D(filters=128, kernel_size=(1,1))(convolution3)
convolution4 = BatchNormalization(axis = -1)(convolution4)
convolution4 = Activation("relu")(convolution4)

pooling2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(convolution4)

convolution5 = Conv2D(filters=256, kernel_size=(1,1))(pooling2)
convolution5 = BatchNormalization(axis = -1)(convolution5)
convolution5 = Activation("relu")(convolution5)

convolution6 = Conv2D(filters=256, kernel_size=(1,1))(convolution5)
convolution6 = BatchNormalization(axis = -1)(convolution6)
convolution6 = Activation("relu")(convolution6)

convolution7 = Conv2D(filters=256, kernel_size=(1,1))(convolution6)
convolution7 = BatchNormalization(axis = -1)(convolution7)
convolution7 = Activation("relu")(convolution7)

pooling3 = MaxPooling2D(pool_size=(2,1), strides=(2,1))(convolution7)

convolution8 = Conv2D(filters=512, kernel_size=(1,1))(pooling3)
convolution8 = BatchNormalization(axis = -1)(convolution8)
convolution8 = Activation("relu")(convolution8)

convolution9 = Conv2D(filters=512, kernel_size=(1,1))(convolution8)
convolution9 = BatchNormalization(axis = -1)(convolution9)
convolution9 = Activation("relu")(convolution9)

convolution10 = Conv2D(filters=512, kernel_size=(1,1))(convolution9)
convolution10= BatchNormalization(axis = -1)(convolution10)
convolution10 = Activation("relu")(convolution10)

pooling4 = MaxPooling2D(pool_size=(2,1), strides=(2,1))(convolution10)

convolution11 = Conv2D(filters=512, kernel_size=(1,1))(pooling4)
convolution11= BatchNormalization(axis = -1)(convolution11)
convolution11 = Activation("relu")(convolution11)

convolution12 = Conv2D(filters=512, kernel_size=(1,1))(convolution11)
convolution12= BatchNormalization(axis = -1)(convolution12)
convolution12 = Activation("relu")(convolution12)

convolution13 = Conv2D(filters=512, kernel_size=(1,1))(convolution12)
convolution13= BatchNormalization(axis = -1)(convolution13)
convolution13 = Activation("relu")(convolution13)

pooling5 = MaxPooling2D(pool_size=(2,1), strides=(2,1))(convolution13)

convolution_full = Reshape(target_shape=(LastFilters * FV, 16))(pooling5)

bidir_LSTM1 = Bidirectional(LSTM(units = NUnits, return_sequences=True))(convolution_full)
bidir_LSTM2 = Bidirectional(LSTM(units = NUnits, return_sequences=True))(bidir_LSTM1)
bidir_LSTM3 = Bidirectional(LSTM(units = NUnits))(bidir_LSTM2)

Model(inputs = input_data, outputs = bidir_LSTM3).summary()

#%%
