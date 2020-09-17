# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:46:37 2020

@author: shashi
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pandas as pd

bank=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on logistic regreession\\bank-full.csv",sep=";")
bank
print(bank.shape)
bank.dropna()
print(bank.shape)
print(bank.isnull().sum())
sn.countplot(x="y",data=bank,palette="hls")
#(Summay of data¶
#Categorical Variables :
#[1] job : admin,technician, services, management, retired, blue-collar, unemployed, entrepreneur, housemaid, unknown, self-employed, student
#[2] marital : married, single, divorced
#[3] education: secondary, tertiary, primary, unknown
#[4] y : yes, no
#[5] housing : yes, no
#[6] loan : yes, no
#[7] deposit : yes, no (Dependent Variable)
#[8] contact : unknown, cellular, telephone
#[9] month : jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
#[10] poutcome: unknown, other, failure, success

#Numerical Variables:
#[1] age[2] balance[3] day[4] duration[5] campaign[6] pdays[7] previous)



## mapping the target variable values.

bank['Y'] = bank['y'].map( {'yes':1, 'no':0} )

bank.drop('y', axis=1,inplace = True)

bank

from pandas_profiling import ProfileReport
report=ProfileReport(bank,explorative=True)
report.to_widgets()
#looking for the impact of the varaiable to each other.
bank.groupby("Y").mean()
#The average age of customers who bought the term deposit is higher than that of the customers who didn’t.
#The pdays (days since the customer was last contacted) is understandably higher for the customers who bought it. The lower the pdays, the better the memory of the last call and hence the better chances of a sale.
#Surprisingly, campaigns (number of contacts or calls made during the current campaign) are lower for customers who bought the term deposit.

bank.groupby("job").mean()
bank.groupby("marital").mean()

## visualization
pd.crosstab(bank.job,bank.Y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

##The frequency of purchase of the deposit depends a great deal on the job title. Thus, the job title can be a good predictor of the outcome variable.
pd.crosstab(bank.marital,bank.Y).plot(kind="bar")

plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')


# MARITAL STATUS DOESN'T SEEMSTO BE  a good predictor.
pd.crosstab(bank.education,bank.Y).plot(kind="bar")
plt.title("bar cahrt of education vs purchase")
plt.xlabel("education")
plt.ylabel("proportion of customers")
#EDUCATION  seems to be  A GOOD PREDICTOR FOR THE DEPENDENT VARIABLE.



pd.crosstab(bank.month,bank.Y).plot(kind="bar")
plt.title("chart of month vs purcahse")
plt.xlabel("month")
plt.ylabel("proportion of customers")

## MONTH DOESNT seems to be a good predictor.

#MAPPING THE VARIABLES WITH YES OR NO VALUES.
bank["default1"]=bank["default"].map( {'yes':1, 'no':0} )
bank.drop("default",axis=1,inplace=True)
bank


# Drop 'contact', as every participant has been contacted. 
bank.drop('contact', axis=1, inplace=True)

# values for "housing" : yes/no
bank["HOUSING"]=bank['housing'].map({'yes':1, 'no':0})
bank.drop('housing', axis=1,inplace = True)


# DropING 'month' and 'day' as they don't have any intrinsic meaning
bank.drop('month', axis=1, inplace=True)
bank.drop('day', axis=1, inplace=True)

bank['recent_pdays'] = np.where(bank['pdays'], 1/bank.pdays, 1/bank.pdays)

# Drop 'pdays'
bank.drop('pdays', axis=1, inplace = True)


bank.tail()
## creating dummy variables.
bank.columns
bank=bank[['age', 'job', 'marital', 'education', 'balance', 'loan', 'duration','campaign', 'previous', 'poutcome',  'default1', 'HOUSING','recent_pdays','Y' ]]

bank.columns
bank
x=bank.iloc[:,0:13]
x
y=bank.iloc[:,13]
y

x=pd.get_dummies(x)
x
x.columns

classifier=LogisticRegression()
classifier.fit(x,y)

y_pred = classifier.predict(x)
bank["y_pred"] = y_pred
bank

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y,y_pred)
confusion_matrix

accuracy = sum(y==y_pred)/bank.shape[0]
accuracy ##accuracy= 0.9003339895158258
pd.crosstab(y_pred,y)


## model accuracy is 90%








