# -*- coding: utf-8 -*-
"""
Created on Mon May  4 09:55:08 2020

@author: shashi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

#loADING THE DATA
zoo=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on knn\\Zoo.csv")
zoo

pd.set_option("display.max_rows",None)
zoo
pd.set_option("display.max_columns",None)
zoo
#CHECHING FOR THE NULL VALUES IF ANY.
zoo.isnull().sum()

zoo.info()

zoo.describe()
#DROPPING THE FIRST COLUMN WHICH IS HAVING THE CATEGORICAL VALUES.
zoo.drop(zoo.columns[[0]],axis=1,inplace=True)
zoo

zoo.type.unique()

#VISUALIZING THE TYPE OF ANIMALS USING THE HISTOGRAM.
plt.hist(zoo.type,bins=7)
#PLOTING TO WHICH TYPE MOST ANIMAL BELONGS TO.
sn.factorplot('type', data=zoo,kind="count", aspect=2)
# from the plot we can see the type 1animals are the most in the no.

#checking for the corelation

cor=zoo.corr()
cor

#now plotting the corelation using heatmap.

plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")

sn.heatmap(cor, annot=True,
            xticklabels=cor.columns.values,
            yticklabels=cor.columns.values)
#checking which variable is having more than 0.7 value.whether positive or negative .
cor[cor != 1][abs(cor)> 0.7]

#checking every variable impact.

zoo.groupby("type").mean()
# transforming in to train and test data.
from sklearn.model_selection import train_test_split
x=zoo.iloc[:,0:16]
x
y=zoo.iloc[:,16]
y
#splitting the data.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)


#building the model.

trn_score=[]
tst_score=[]
for i in range(1,40):
    knn=KNC(i)
    knn.fit(x_train,y_train)
    trn_score.append(knn.score(x_train,y_train))
    tst_score.append(knn.score(x_test,y_test))    



max_trn_score=max(trn_score)
trn_score_i= [i for i, v in enumerate(trn_score) if v == max_trn_score]
print('max train score {}% and k={}'.format(max_trn_score*100,list(map(lambda x: x+1, trn_score_i))))

#max train score is 100% AT K=1
max_tst_score=max(tst_score)
tst_score_i= [i for i, v in enumerate(tst_score) if v == max_tst_score]
print('max tst score {}% and k={}'.format(max_tst_score*100,list(map(lambda x: x+1, tst_score_i))))
# maxx test accurcay is 100% at k=1,2,3.

plt.figure(figsize=(10,5))
sn.lineplot(range(1,40),trn_score,marker="*",label="train_score")
sn.lineplot(range(1,40),tst_score,marker="+",label="test_score")
y_pred=knn.predict(x_test)
cm1=confusion_matrix(y_test,y_pred)
cm1

#by plotting the accuracy we analysed that by increasing the value of k after k=3 the accuracy is getting decreased.
# checking for the various variables which are having greater corelation..
# hair is having highest corelation values.
sn.countplot(x="hair", data=zoo)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()
zoo.loc[:,'hair'].value_counts()

#by anlysing this most of the time hair is o means no impact so removing this variavle.

zoo.drop(zoo.columns[[0]],axis=1,inplace=True)
zoo

x1=zoo.iloc[:,0:15]
x1
y1=zoo.iloc[:,15]
y1

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=1,stratify=y)

trn1_score=[]
tst1_score=[]
for i in range(1,40):
    knn1=KNC(i)
    knn1.fit(x1_train,y1_train)
    trn1_score.append(knn1.score(x1_train,y1_train))
    tst1_score.append(knn1.score(x1_test,y1_test))    



max_trn1_score=max(trn1_score)
trn1_score_i= [i for i, v in enumerate(trn1_score) if v == max_trn1_score]
print('max train score {}% and k={}'.format(max_trn1_score*100,list(map(lambda x: x+1, trn1_score_i))))

#max train score is 100% AT K=1
max_tst1_score=max(tst1_score)
tst1_score_i= [i for i, v in enumerate(tst1_score) if v == max_tst1_score]
print('max tst score {}% and k={}'.format(max_tst1_score*100,list(map(lambda x: x+1, tst1_score_i))))

#max test score is 100%.
# type1 animals are 41
#type2 animals are 20
#type3 animals are 5
#type4 animals are 13
#type5 animals are 4
#type6 animals are 8
#type7 animals are 10

#now by analysing all the accuracy score we are considering k=3 nearest neighbour.

