# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:37:51 2019

@author: Patryk
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

model = Sequential()

model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

model.fit(data, labels, epochs=10, batch_size=32)

model.summary()