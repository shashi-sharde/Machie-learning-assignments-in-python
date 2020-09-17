# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:33:50 2020

@author: shashi
"""


import numpy as np
import pandas as pd

data = pd.read_csv('C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on support vector machine\\forestfires.csv')
data.info()
data.head()
#no null values 




E = data.drop(['month','day'], axis = 1)

X = E.drop('size_category',axis =1)
Y = E[['size_category']]

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.33)

from sklearn.svm import SVC
#linear model
model = SVC(kernel = 'linear')
model.fit(train_x, train_y)
pred = model.predict(test_x)
model.score(test_x,test_y)  # accuracy = 0.98
#poly model

model1 = SVC(kernel='poly')
model1.fit(train_x,train_y)
model1.predict(test_x)
model1.score(test_x,test_y)# accuracy = 0.96

#rbf model 

model2 = SVC(kernel = 'rbf')
model2.fit(train_x,train_y)
model2.predict(test_x)
model2.score(test_x,test_y)# accuracy = 0.73

#hence kernel = linear , gives us maximum accuracy of 98 percent 
