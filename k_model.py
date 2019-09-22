# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:31:58 2019

@author: Patryk
"""

from keras.models import Model
from keras.layers import Dense, Input

input = Input(shape=(100,))
L1 = Dense(10, activation='relu')(input)
L2 = Dense(20, activation='relu')(L1)
output = Dense(1, activation='softmax')(L2)

model = Model(inputs=input, outputs=output)

model.summary()