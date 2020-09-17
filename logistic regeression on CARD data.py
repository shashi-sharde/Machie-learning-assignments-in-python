# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:04:44 2020

@author: shashi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

card=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on logistic regreession\\creditcard.csv")
card

card.drop(card.columns[[0]],axis=1,inplace=True)
card

sn.countplot(x="card",data=card,palette="hls")

card.isnull().sum()
## no null values
sn.countplot(x="card",data=card,palette="hls")
pd.crosstab(card.reports,card.card).plot(kind="bar")
pd.crosstab(card.age,card.card).plot(kind="bar")
pd.crosstab(card.income,card.card).plot(kind="bar")
pd.crosstab(card.active,card.card).plot(kind="bar")
## analysing all the independent variables impact on the dependent variable.
sn.pairplot(card)

card.corr()


mapping={"yes":1,"no":0}

for col in card:
    if card[col].dtypes == object:
        card[col]=card[col].map(mapping)

card

X = card.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
X
Y = card.iloc[:,0]
Y
## creating model
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_

y_pred = classifier.predict(X)
card["y_pred"] = y_pred
card

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
confusion_matrix

accuracy = sum(Y==y_pred)/card.shape[0]
accuracy ##0.98
pd.crosstab(y_pred,Y)

#BY ANALYSING THE CROSSTABLE OF PREDICTION AND ACTUAL OUR MODEL HAS ACCURATELY PREDICTED "NO"  AS "NO" 295 TIMES.AND ACCURATELY PREDICTED "YES" AS "YES" 1000 TIMES.

## AND WRONGLY PREDICTED "NO" AS "YES"23 TIMES. AND WRONGLY PREDICTED "YES"AS NO

##IT HAS THE ACCURACY OF 0.98