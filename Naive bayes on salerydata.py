#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries i required to envoke
import pandas as pd
import numpy as np
from sklearn. naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# In[2]:


#reading the  train dataset
df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on naive base\\SalaryData_Train.csv")
df


# In[3]:


#Doing the eda
#required libray
from pandas_profiling import ProfileReport


# In[4]:


profile=ProfileReport(df,title='profile report of the data',explorative=True)


# In[5]:


profile.to_widgets()


# In[6]:


# AFTER EDA I HAVE ANALYSED THAT THE DATASET IS HAVING TWO MANY DUPLICATE VALUES SO I NEED TO REMOVE THEM
#REMOVING THE DUPLICATE VALUES
df=df.drop_duplicates()
df


# In[7]:


#checking whether the duplicated values are removed or not.
a=df.duplicated()
a


# In[ ]:





# In[8]:


#there are no  missing values in the data
# the are no multicolinearity between the data because no variable is showing highly correlated with one another.
# so i analysed that various columns are not giving any significant impact on the output so removig the columns .
#then we will be spliting the data in to train snd test.
#then we will be building the two model one is gaussian classifier and another one is multinomialclassifier


# In[9]:


df.drop(df.columns[[1,2,5,6,7,9,10,11]], axis=1,inplace=True)
df


# In[10]:


#convering the variable maritalstatus in to dummy variable
dummies = pd.get_dummies(df['maritalstatus']).rename(columns=lambda x: 'maritalstatus_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable maritalstatus from the dataset after the creatin of its dummies.
df=df.drop('maritalstatus',1)
df


# In[11]:


#convering the variable sex in to dummy variable
dummies = pd.get_dummies(df['sex']).rename(columns=lambda x: 'sex_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable sex from the dataset after the creatin of its dummies.
df=df.drop('sex',1)
df


# In[12]:



df=df.drop('native',1)
df


# In[13]:


predictors=df.drop(['Salary'],axis=1)
predictors


# In[14]:


ip_columns = ['age','educationno','sex_ Female','sex_ Male','maritalstatus_ Divorced','maritalstatus_ Married-AF-spouse','maritalstatus_ Married-civ-spouse','maritalstatus_ Married-spouse-absent','maritalstatus_ Never-married','maritalstatus_ Separated','maritalstatus_ Widowed']
op_column = ['Salary']


# In[15]:


#splitig the data
Xtrain,Xtest,ytrain,ytest = train_test_split(df[ip_columns],df[op_column],test_size=0.3, random_state=0)


# In[16]:


ignb = GaussianNB()
imnb = MultinomialNB()


# In[17]:



# Building and predicting at the same time

pred_gnb = ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)


# In[18]:


# Confusion matrix GaussianNB model
confusion_matrix(ytest,pred_gnb) # GaussianNB model
pd.crosstab(ytest.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==ytest.values.flatten()) #giving the accurCY OF 72.63%


# In[19]:


# Confusion matrix multinomialNB model
confusion_matrix(ytest,pred_mnb) # multinomialNB model
pd.crosstab(ytest.values.flatten(),pred_mnb) # confusion matrix using 
np.mean(pred_mnb==ytest.values.flatten()) # 72.99%


# In[20]:


confusion_matrix(ytest,pred_mnb) # multinomailNB model


# In[21]:


confusion_matrix(ytest,pred_gnb) # GaussianNB model


# In[ ]:




