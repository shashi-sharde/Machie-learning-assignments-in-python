#!/usr/bin/env python
# coding: utf-8

# In[1]:


#impoerting the required libraries.
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


#loading the dataset.
df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on decision tree\\Company_Data.csv")
df


# In[3]:


#eda
profile=ProfileReport(df,title='profile report of the data',explorative=True)


# In[4]:


profile.to_widgets()


# In[ ]:


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


# In[5]:


#convering the variable urban in to dummy variable
dummies = pd.get_dummies(df['Urban']).rename(columns=lambda x: 'Urban_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable urban from the dataset after the creatin of its dummies.
df=df.drop('Urban',1)
df


# In[6]:


#now converting the US variable in to its dummies.
dummies=pd.get_dummies(df['US']).rename(columns=lambda x: "US_" +str(x))
#Bringing the dummies in to original dataset
df=pd.concat([df,dummies],axis=1)
#Dropping the variable US from the dataset
df=df.drop("US",1)
df


# In[7]:


# Creting the dummies for the shelveloc variable
dummies=pd.get_dummies(df['ShelveLoc']).rename(columns=lambda x: "ShelvoLoc_"+str(x))
#bringing the dummies of shelveloc in to the main dataset
df=pd.concat([df,dummies],axis=1)
#dropping the column ShelveLoc from the dataset
df=df.drop("ShelveLoc",1)
df


# In[48]:


# now converting the sales data in to different category
row_indexes=df[df['Sales']>=8].index
df.loc[row_indexes,'sales']="yes"
row_indexes=df[df['Sales']<8].index
df.loc[row_indexes,'sales']="no"
#now dropping the column Sales
df=df.drop("Sales",1)
df

#AT THIS POINT WE HAVE DONE ALL THE PREPROCESSING REQUIRED FOR THE DATA.


# In[10]:


#creating the variable containg all the column'
columns=list(df.columns)
columns


# In[35]:


X=columns[:14]
Y=columns[14]
print(X)
Y


# In[20]:


#spliting the dataset in to train and test
train,test=train_test_split(df,test_size=0.20)


# In[21]:


#creating the model
Dtree=DecisionTreeClassifier(criterion="entropy")
Dtree.fit(train[X],train[Y])


# In[22]:


#evaluating the model by predicting the result.
pred=Dtree.predict(test[X])
pred


# In[ ]:





# In[23]:


#  value count of predicted values on the test data.
pd.Series(pred).value_counts()


# In[24]:


#creating the cross table for the predicted vales and the actual value
pd.crosstab(test[Y],pred)


# In[25]:



#checkin the accuracy of the model
np.mean(pred==test.sales)


# In[ ]:


#its giving the accuracy of 77.5%.

