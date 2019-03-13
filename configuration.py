# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:55:33 2019

@author: Gireesh Sundaram
"""

#General configurations
window_height = 64  #windown height
window_width = 64   #window width
window_shift = window_width - 2 #window shift

#CNN related configurations
MPoolLayers_ALL = 5	#Nbr of all maxpool layers
MPoolLayers_H = 2	#Nbr of maxpool in horizontal dimension
LastFilters = 512	#Nbr of feature maps at the last conv layer

#LSTM related configurations
NUnits = 256    #Number of units in forward/backward LSTM
NLayers = 3     #Number of layers in BLSTM