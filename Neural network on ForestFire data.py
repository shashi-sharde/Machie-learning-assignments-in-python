# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:43:58 2020

@author: shashi
"""


import pandas as pd
fire = pd.read_csv('C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignmetn on neural networks\\forestfires (1).csv')
fire.head()

E = fire.drop(['month','day'], axis = 1)
#droping month and day column
mapping = {"small":0, "large":1}
for col in E:
    if E[col].dtypes == object:
        E[col] = E[col].map(mapping)
#mapping size_catagory column 
E        
X = E.drop('size_category',axis =1)
#X is our input variable
Y = E[['size_category']]
# Y is our target variable

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.33)
model.add(Dense(80, activation ='relu', input_dim = 9))
model.add(Dense(70, activation = 'relu'))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(1, kernel_initializer ='normal' ))
model.compile(optimizer ='adam', loss = 'mean_squared_error', metrics =['mse'])
model.fit(train_x,train_y, validation_data = (test_x,test_y), epochs = 120)


train_pred = model.predict(train_x)
test_pred = model.predict(test_x)

from sklearn.metrics import r2_score

train_r2 = r2_score(train_y, train_pred)
test_r2 = r2_score(test_y,test_pred)
#we got model accuracy of 86.45 percent 