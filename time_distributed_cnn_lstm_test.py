# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:54:26 2019

@author: tornado
"""

import pandas as pd

from keras.layers import Input, TimeDistributed, Bidirectional, Conv2D, BatchNormalization, MaxPooling2D, Flatten, LSTM, Dense, Lambda, GRU, Activation
from keras.optimizers import SGD
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.merge import add, concatenate

import keras.backend as K
from time_distributed_read_images import generate_arrays_from_file

from config import train_file, window_height, window_width, channels, stride, max_image_width, no_of_frames

from matplotlib import pyplot as plt

from time_distributed_read_images import labels_to_text, image_out

import numpy as np
import editdistance

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#%%
input_data = Input(shape= (no_of_frames, window_height, window_width, channels), name= "the_input")

conv = TimeDistributed(Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv1")(input_data)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm1")(conv)
conv = TimeDistributed(Conv2D(filters=64, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv2")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm2")(conv)
pooling = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name = "max_pool1")(bn)

conv = TimeDistributed(Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv3")(pooling)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm3")(conv)
conv = TimeDistributed(Conv2D(filters=128, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv4")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm4")(conv)
pooling = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name = "max_pool2")(bn)

conv = TimeDistributed(Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv5")(pooling)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm5")(conv)
conv = TimeDistributed(Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv6")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm6")(conv)
conv = TimeDistributed(Conv2D(filters=256, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv7")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm7")(conv)
pooling = TimeDistributed(MaxPooling2D(pool_size=(2,1)), name = "max_pool3")(bn)

conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv8")(pooling)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm8")(conv)
conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv9")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm9")(conv)
conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv10")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm10")(conv)
pooling = TimeDistributed(MaxPooling2D(pool_size=(2,1)), name = "max_pool4")(bn)

conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv11")(pooling)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm11")(conv)
conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv12")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm12")(conv)
conv = TimeDistributed(Conv2D(filters=512, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal'), name = "conv13")(bn)
bn = TimeDistributed(BatchNormalization(), name = "batch_norm13")(conv)
pooling = TimeDistributed(MaxPooling2D(pool_size=(2,1)), name = "max_pool5")(bn)

flatten = TimeDistributed(Flatten(), name = "flatten")(pooling)

dense = TimeDistributed(Dense(128), name = "dense")(flatten)

blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25), name = "BiLSTM1")(dense)
blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25), name = "BiLSTM2")(blstm)
blstm = Bidirectional(LSTM(256, return_sequences=True, dropout=0.25), name = "BiLSTM3")(blstm)

dense = TimeDistributed(Dense(90 + 1, name="output"))(blstm)
y_pred = Activation('softmax', name='softmax')(dense)

Model(inputs = input_data, outputs = y_pred).summary()

#%%
filepath="Checkpoints/td.weights.last.hdf5"

model_p = Model(inputs=input_data, outputs=y_pred)
model_p.load_weights(filepath)

def decode_predict_ctc(out, input_length, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=input_length,
                           greedy=False, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = labels_to_text(lables)
      results.append(text)
    return results
  
def predit_a_image(a, top_paths = 1):
  c = np.expand_dims(a.T, axis=0)
  net_out_value = model_p.predict(c)
  top_pred_texts = decode_predict_ctc(net_out_value, top_paths)
  return top_pred_texts

#%%
training = pd.read_csv(train_file)
training["Required"] = training["Annotation"].str.len() #Length of the annotation
training["Available"] = (((training["Width"] // window_width) + 1) * window_height // stride - stride + 1 ) //2
bad_samples = training[training["Required"] + 3 > training["Available"]]
training = training[training["Required"] + 3 < training["Available"]]
training = training[training["Width"] <= max_image_width]
training = training.dropna().reset_index().drop(columns = ["index"])

#%%
error = 0
for i in range(0, len(training)):
    a, input_length = image_out(training, training["Path"][i])
    net_out_value = model_p.predict(a)
    pred_texts = decode_predict_ctc(net_out_value, input_length[0])
    print(pred_texts, " -- ", str(i + 1))
    print(training["Annotation"][i])    
    error = error + editdistance.eval(pred_texts, training["Annotation"][i]) / len(training["Annotation"][i])
    per_error = error / (i+1)
    print(error)
    print(per_error)
    print("--------------------------")