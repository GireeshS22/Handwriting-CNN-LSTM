# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:08:57 2019

@author: Gireesh Sundaram
"""


import os

#%%
path = "D:\CBA\Capstone\Handwriting_DeepCRNN\Data\Images\Test\\"

for r, d, f in os.walk(path):
    for file in f:
        if ".png" in file:
            os.rename(path + file, path + file.upper())
        if ".PNG" in file:
            print(file)