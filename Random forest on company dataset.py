#!/usr/bin/env python
# coding: utf-8

# In[1]:


#impoerting the required libraries.
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\Assignment on rndome forest\\Company_Data.csv")
df


# In[3]:


#eda
profile=ProfileReport(df,title='profile report of the data',explorative=True)


# In[4]:


profile.to_widgets()


# In[5]:


# POINTS  BELOW ARE THE CONCLUSIONS OF THE EDA I HAVE DONE.
#1) the dataset is containg the 11 variables,in which '8' variables are the numeric ,'2' variables are the boolean type and  variable is categorical type.
#2) There are no missing columns or values in our dataset.
#3) there are total 400 observation.
#4) There are no high corelation between the variables respect to one another.

# operations to Follow
#1) we need to convert the Variables which are in boolean type and categorical type in to dummy variables.
#2) We need to convert the target variable which is sale in to categorical variable type.
#3) Then model building
#4) The model evaluation.


# In[6]:


#now converting the US variable in to its dummies.
dummies=pd.get_dummies(df['US']).rename(columns=lambda x: "US_" +str(x))
#Bringing the dummies in to original dataset
df=pd.concat([df,dummies],axis=1)
#Dropping the variable US from the dataset
df=df.drop("US",1)
df
#now converting the US variable in to its dummies.
dummies=pd.get_dummies(df['Urban']).rename(columns=lambda x: "Urban_" +str(x))
#Bringing the dummies in to original dataset
df=pd.concat([df,dummies],axis=1)
#Dropping the variable US from the dataset
df=df.drop("Urban",1)
df


# In[7]:


# Creting the dummies for the shelveloc variable
dummies=pd.get_dummies(df['ShelveLoc']).rename(columns=lambda x: "ShelvoLoc_"+str(x))
#bringing the dummies of shelveloc in to the main dataset
df=pd.concat([df,dummies],axis=1)
#dropping the column ShelveLoc from the dataset
df=df.drop("ShelveLoc",1)
df


# In[8]:


# now converting the sales data in to different category
row_indexes=df[df['Sales']>=8].index
df.loc[row_indexes,'sales']="yes"
row_indexes=df[df['Sales']<8].index
df.loc[row_indexes,'sales']="no"
#now dropping the column Sales
df=df.drop("Sales",1)
df

#AT THIS POINT WE HAVE DONE ALL THE PREPROCESSING REQUIRED FOR THE DATA.


# In[9]:


#creating the variable containg all the column'
columns=list(df.columns)
columns


# In[10]:


X=columns[:14]
Y=columns[14]
print(X)
Y


# In[11]:


#spliting the dataset in to train and test
train,test=train_test_split(df,test_size=0.20)


# In[12]:


#Building the model
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")


# In[13]:


rf.fit(train[X],train[Y]) # Fitting RandomForestClassifier model from sklearn.ensemble 


# In[14]:


rf.estimators_ 


# In[15]:


rf.classes_ # class labels (output)


# In[16]:



rf.n_classes_ # Number of levels in class labels 


# In[17]:


rf.n_features_  # Number of input features in model 8 here.


# In[18]:


rf.n_outputs_ # Number of outputs when fit performed


# In[19]:


pred=rf.predict(test[X])
pred


# In[20]:


#  value count of predicted values on the test data.
pd.Series(pred).value_counts()


# In[21]:


#creating the cross table for the predicted vales and the actual value
pd.crosstab(test[Y],pred)


# In[22]:


#checkin the accuracy of the model
np.mean(pred==test.sales) #it is giving the accuracy of 83.75%

