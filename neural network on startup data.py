# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:11:33 2020

@author: shashi
"""

import pandas as pd
start = pd.read_csv('C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignmetn on neural networks\\50_Startups.csv')
start.head()

E = start.drop('State', axis = 1)
E.head()

X = E.drop('Profit', axis = 1)
X.head()
Y = E[['Profit']]
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
train_x,test_x,train_y,test_y =train_test_split(X,Y, test_size = 0.33)
# building model 1 
model = Sequential()

model.add(Dense(80, activation ='relu', input_dim = 3))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'normal'))
model.compile(optimizer ='adam', loss = 'mean_squared_error', metrics =['mse'])

model.fit(train_x,train_y, validation_data = (test_x,test_y), epochs = 60)

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

from sklearn.metrics import r2_score

train_r2 = r2_score(train_y, train_pred)
test_r2 = r2_score(test_y,test_pred)# 0.903
#hence this model gives us 90 percent accuracy