# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:44:41 2019

@author: Gireesh Sundaram
"""

import cv2
import numpy as np
import pandas as pd
import codecs
import math

from config import window_height, window_width, nb_labels, stride, max_image_width, max_label_len, no_of_frames, mini_batch_size

from keras.preprocessing import sequence

#%%
#reading the class files
data = {}
with codecs.open("Data/class.txt", 'r', encoding='utf-8') as cF:
    data = cF.read().split('\n')
    
#%%
def returnClasses(string):
    text = list(string)
    text = ["<SPACE>"] + ["<SPACE>" if x==" " else x for x in text] + ["<SPACE>"]
    classes = [data.index(x) if x in data else 2 for x in text]
    classes = np.asarray(classes)
    return classes
    
#%%
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(data):  # CTC Blank
            ret.append("")
        else:
            ret.append(data[c])
    output = "".join(ret)
    output = output.replace("<SPACE>", " ").strip()
    return output

#%%
def split_frames(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    h, w = np.shape(image)
#    print(np.shape(image))
    
    if (h > window_height): factor = window_height/float(h)
    else: factor = 1
    
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h, w = np.shape(image)
    
    if (h < window_height): image = cv2.resize(image, (w, window_height))
    
    h, w = np.shape(image)
#    print(np.shape(image))    
    
    if w / window_width < math.ceil(w / window_width):
        padding = np.full((window_height, math.ceil(w / window_width) * window_width - w), 255)
        image = np.append(image, padding, axis = 1)
#        print("++", str(np.shape(image)))
    
    h, w = np.shape(image)
    frames = np.full((no_of_frames, window_height, window_width, 1), 255)
    
    slide = window_width
    i = 0
    
    while slide <= w:
        end = slide
        start = end - window_width
#        print("Start and end are: ", str(start), str(end))
        this_frame = image[:, start:end]
        this_frame = np.expand_dims(this_frame, 2)
        frames[i] = this_frame
        slide = slide + stride
        i = i + 1
                
    return frames, i

#%%
def generate_arrays_from_file(infile):
    while True:
#        infile = pd.read_excel(path)
        for record in range(0, len(infile)):
            print("Reading file: " + str (record))
            path = infile["Path"][record]
            annotation = infile["Annotation"][record]
            print(annotation)

            no_of_frames = ((max_image_width // window_width) + 1) * 64 // 8 - 8 + 1            
            image_out = np.zeros((1, no_of_frames, window_height, window_width, 1))
            image_out[0], width = split_frames(path)
            
#            print(np.shape(image_out))
            
            label_out = np.zeros((1, max_label_len))
            annot = returnClasses(annotation)
            if len(annot) < max_label_len:
                padding = np.full((max_label_len - len(annot)), 91)
                annot = np.append(annot, padding, axis = 0)
            label_out[0] = annot
#            print(np.shape(label_out))
            
            input_length = np.zeros([1, 1])
            input_length[0] = width
            
#            print(input_length)
            
            label_length = np.zeros((1, 1), dtype=np.float32)
            
            label_length[0] = len(annotation)
            
            inputs = {
                    'the_input': image_out,     #Batch Size, h, w, no of channels
                    'the_labels': label_out,    #Batch size, max len of labels - 91 for labels that are over the actual label
                    'input_length': input_length,   #Batch size, 1
                    'label_length': label_length    #Batch size, 1
                    }
            outputs = {'ctc': np.zeros([1])}    #Batch size, 1
            
            yield(inputs, outputs)
            
#%%
def generate_val_arrays_from_file(infile):
    while True:
#        infile = pd.read_excel(path)
        for record in range(0, len(infile)):
#            print("Reading file: " + str (record))
            path = infile["Path"][record]
            annotation = infile["Annotation"][record]
#            print(annotation)

            image_out = np.zeros((1, no_of_frames, window_height, window_width, 1))
            image_out[0], width = split_frames(path)
            
#            print(np.shape(image_out))
            
            label_out = np.zeros((1, max_label_len))
            annot = returnClasses(annotation)
            if len(annot) < max_label_len:
                padding = np.full((max_label_len - len(annot)), 91)
                annot = np.append(annot, padding, axis = 0)
            label_out[0] = annot
#            print(np.shape(label_out))
            
            input_length = np.zeros([1, 1])
            input_length[0] = width
            
#            print(input_length)
            
            label_length = np.zeros((1, 1), dtype=np.float32)
            
            label_length[0] = len(annotation)
            
            inputs = {
                    'the_input': image_out,     #Batch Size, h, w, no of channels
                    'the_labels': label_out,    #Batch size, max len of labels - 91 for labels that are over the actual label
                    'input_length': input_length,   #Batch size, 1
                    'label_length': label_length    #Batch size, 1
                    }
            outputs = {'ctc': np.zeros([1])}    #Batch size, 1
            
            yield(inputs, outputs)
            
#%%
def image_out(infile, path):
    image_out = np.zeros((1, no_of_frames, window_height, window_width, 1))
    image_out[0], width = split_frames(path)
    
    input_length = np.zeros([1, 1])
    input_length[0] = width
    
    return image_out, input_length