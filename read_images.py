# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:44:41 2019

@author: Gireesh Sundaram
"""

import cv2
import numpy as np
import math
import pandas as pd
import codecs

from configuration import window_height, window_width, window_shift, MPoolLayers_H

#%%
vec_per_window = window_width / math.pow(2, MPoolLayers_H)

#%%
#reading the class files
data = {}
with codecs.open("Data\\class.txt", 'r', encoding='utf-8') as cF:
    data = cF.read().split('\n')
    
#%%
def returnClasses(string):
    text = list(string)
    text = ["<SPACE>"] + ["<SPACE>" if x==" " else x for x in text] + ["<SPACE>"]
    classes = [data.index(x) if x in data else 2 for x in text]
    return classes
    
#%%
infile = pd.read_csv("Data\\list.csv")

#%%
#reading images from the path
image = cv2.imread("Data\\Images\\Test\\R06-137-S00-03.PNG", cv2.IMREAD_GRAYSCALE)
h, w = np.shape(image)

#setting factor if the image is greater than the window height
if (h > window_height): factor = window_height/h
else: factor = 1

#resizing the image to the specified window height
image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
h, w = np.shape(image)

#writing to input list
inputList = []
seqLens = []

featureSet = []

winId = 0
while True:
    s = winId * window_shift
    e = s + window_width
    
    if e > w:
        break
    
    wnd = image[:h, s:e]
    featureSet.append(wnd)
    winId = winId + 1
    
inputList.append(featureSet)

#taking the sequence length
winId = 0
wpd = 0
while True:
    s = winId * window_shift
    e = s + window_width
    
    if e > w:
        sl = (winId+1) * vec_per_window
        seqLens.append(sl)
        break
    winId = winId + 1
    
#taking the text annotations from the csv file and converting to labels
    