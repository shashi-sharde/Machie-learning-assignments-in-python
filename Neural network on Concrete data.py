# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:23:07 2020

@author: shashi
"""


import pandas as pd
con = pd.read_csv('C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignmetn on neural networks\\concrete.csv')
con.head()

X = con.drop('strength', axis = 1)
X.head()

Y = con[['strength']]
Y.head()
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
train_x,test_x,train_y,test_y =train_test_split(X,Y, test_size = 0.33)
# building model 1 
model = Sequential()
n_cols = train_X.shape[1]

model.add(Dense(50, activation ='relu', input_dim = 8))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'normal'))
model.compile(optimizer ='adam', loss = 'mean_squared_error', metrics =['mse'])

from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience = 3)

import numpy as np
model.fit(train_x,train_y, validation_data = (test_x,test_y), epochs = 10)
callbacks = [early_stopping_monitor]

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

from sklearn.metrics import r2_score
train_r2 = r2_score(train_y, train_pred)
test_r2 = r2_score(test_y, test_pred)# acuracy = 0.79

#building second model 
model2 = Sequential()

model2.add(Dense(100,activation = 'relu', input_dim = 8))
model2.add(Dense(100,activation ='relu'))
model2.add(Dense(80,activation = 'relu'))
model2.add(Dense(1,kernel_initializer = 'normal'))

model2.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse'])

model2.fit(train_x,train_y, validation_data = (test_x,test_y), epochs = 30)

train2_pred = model2.predict(train_x)
test2_pred = model2.predict(test_x)

train2_r2 = r2_score(train_y, train_pred)
