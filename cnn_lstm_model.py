# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:23:55 2019

@author: Gireesh Sundaram
"""

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Reshape
from keras.layers import Bidirectional, LSTM, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

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
y_pred = Bidirectional(LSTM(units = NUnits))(bidir_LSTM2)

Model(inputs = input_data, outputs = y_pred).summary()

#%%
# the actual loss calc occurs here despite it not being an internal Keras loss function

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#%%
labels = Input(shape=[47], dtype='float32')
input_length = Input(shape=[1], dtype='int64')
label_length = Input(shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# clipnorm seems to speeds up convergence
sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)