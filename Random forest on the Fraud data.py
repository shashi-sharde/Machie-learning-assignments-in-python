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


df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\Assignment on rndome forest\\Fraud_check.csv")
df


# In[3]:


#EDA
from pandas_profiling import ProfileReport
report=ProfileReport(df,title="EDA report of the fraud data",explorative=True)


# In[4]:


report.to_widgets()


# In[5]:


## POINTS  BELOW ARE THE CONCLUSIONS OF THE EDA I HAVE DONE.
#1) the dataset is containg the 6 variables,in which '3' variables are the numeric ,'2' variables are the boolean type and  variable is categorical type.
#2) There are no missing columns or values in our dataset.
#3) there are total 600 observation.
#4) There are no high corelation between the variables with respect to each other.

# operations to Follow
#1) we need to convert the Variables which are in boolean type and categorical type in to dummy variables.
#2) We need to convert the target variable which is  taxable income in to categorical variable type.
#3) Then model building
#4) Then model evaluation.


# In[6]:


##creating  the dummy  variable for  urban variable
dummies = pd.get_dummies(df['Urban']).rename(columns=lambda x: 'Urban_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable urban from the dataset after the insertion of its dummies.
df=df.drop('Urban',1)
df


# In[7]:


#creating the  dummy variable for the undergrad variable
dummies = pd.get_dummies(df['Undergrad']).rename(columns=lambda x: 'Undergrad_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable undergrad from the dataset after the insertion of its dummies.
df=df.drop('Undergrad',1)
df


# In[8]:


#creating the dummy variables for the variable Marital.status
dummies = pd.get_dummies(df['Marital.Status']).rename(columns=lambda x: 'Marital.status_' + str(x))
# bring the dummies back into the original dataset
df = pd.concat([df, dummies], axis=1)
#now dropping the variable Marital.Status from the dataset after the insertion of its dummies.
df=df.drop('Marital.Status',1)
df


# In[9]:


#CONVERTING THE VALUES OF TAXABLE.INCOME VARIABLE IN TO DIFFERENT CATEGORIES
row_indexes=df[df['Taxable.Income']>=30000].index
df.loc[row_indexes,'Taxable_income']="GOOD"
row_indexes=df[df['Taxable.Income']<=30000].index
df.loc[row_indexes,'Taxable_income']="RISKY"
#DROPING THE VARIABLE TAXABLE.INCOME VARIABLE AFTER THE INSERTION OF ITS CONVERTED VALUES IN DIFFERENT COLUMN
df=df.drop('Taxable.Income',1)
df


# In[10]:


colms = list(df.columns)
colms


# In[11]:


X = colms[:9]
Y = colms[9]
print(X)
print(Y)


# In[12]:


#splitting the data int to train and test
train,test=train_test_split(df,test_size=0.2)


# In[13]:


#creating the model
rf = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")


# In[14]:


rf.fit(train[X],train[Y]) # Fitting RandomForestClassifier model from sklearn.ensemble 


# In[15]:


rf.estimators_ 


# In[16]:


rf.classes_ # class labels (output)


# In[17]:


rf.n_classes_ # Number of levels in class labels 


# In[18]:


rf.n_features_  # Number of input features in model 8 here.


# In[19]:


rf.n_outputs_ # Number of outputs when fit performed


# In[20]:


pred=rf.predict(test[X])
pred


# In[21]:


#creating the cross table for the predicted vales and the actual value
pd.crosstab(test[Y],pred)


# In[23]:


#checkin the accuracy of the model
np.mean(pred==test.Taxable_income) #it is giving the accuracy of 75%

