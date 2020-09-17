# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 20:58:49 2020

@author: shashi
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on naive base\\SalaryData_Train.csv")
df
df.info()
df.head()
df.duplicated()
#removing the duplicate rows
df=df.drop_duplicates()
df
#checking whether the duplicated values are removed or not.

a=df.duplicated()
a
 
# removing the columns which are not having a significant impact on the target ariable in my analysation.

df.drop(df.columns[[1,2,5,6,7,9,10,11]], axis=1,inplace=True)
df
#convering the variable maritalstatus in to dummy variable
dummies = pd.get_dummies(df['maritalstatus']).rename(columns=lambda x: 'maritalstatus_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable maritalstatus from the dataset after the creatin of its dummies.
df=df.drop('maritalstatus',1)
df

#convering the variable sex in to dummy variable
dummies = pd.get_dummies(df['sex']).rename(columns=lambda x: 'sex_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable sex from the dataset after the creatin of its dummies.
df=df.drop('sex',1)
df

df=df.drop('native',1)
df

predictors=df.drop(['Salary'],axis=1)
predictors

ip_columns = ['age','educationno','sex_ Female','sex_ Male','maritalstatus_ Divorced','maritalstatus_ Married-AF-spouse','maritalstatus_ Married-civ-spouse','maritalstatus_ Married-spouse-absent','maritalstatus_ Never-married','maritalstatus_ Separated','maritalstatus_ Widowed']
op_column = ['Salary']

#splitig the data
Xtrain,Xtest,ytrain,ytest = train_test_split(df[ip_columns],df[op_column],test_size=0.3, random_state=0)


#linear model
model = SVC(kernel = 'linear')
model.fit(Xtrain, ytrain)
pred = model.predict(Xtest)
model.score(Xtest,ytest)  # accuracy = 81.36%
#poly model

model1 = SVC(kernel='poly')
model1.fit(Xtrain, ytrain)
model1.predict(Xtest)
model1.score(Xtest,ytest)# accuracy = 80.90%

#rbf model 

model2 = SVC(kernel = 'rbf')
model2.fit(Xtrain, ytrain)
model2.predict(Xtest)
model2.score(Xtest,ytest)# accuracy = 81.29%




#so i will be selecting the linear model which is giving the highest accuracy.