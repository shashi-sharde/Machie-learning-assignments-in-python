# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 00:26:05 2020

@author: shashi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 

# loading the dataset
gls=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on knn\\glass.csv")
gls
pd.set_option('display.max_rows',None)
gls

##checking for the null values.
gls.isnull().sum()
#no null values.

#geting information about the general values of each feature
gls.describe()
##checking the info of the data.
gls.info()
#checking histogoram of each data.
gls.hist()
#checking the box plot.
a=sn.boxplot(data=gls)
## the plot doesn't seems to be having large diffrence in range.
b=sn.pairplot(data=gls)

x=gls.iloc[:,0:9]
x
y=gls.iloc[:,9]
y
#Get the distribution of the different classifications
y.value_counts().plot(kind="bar")
#We have an unequal distribution of dependent variables.

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5,stratify=y)

#I used "stratify = y." The stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to the "stratify" parameter. In this case, we are using it to ensure there is an equal proportion of labels in our training and testing sets.

train_score=[]
test_score=[]

# applying knn algo for different values of k ranges from 1 to 10.
for i in range(1,11):
    knn=KNC(i)
    knn.fit(x_train,y_train)
    train_score.append(knn.score(x_train,y_train))
    test_score.append(knn.score(x_test,y_test))
## accessing the model.
#and analysing at  which k value the model is giving highest train sscore and test score.

max_train_score=max(train_score)
train_score_i= [i for i, v in enumerate(train_score) if v == max_train_score]
print('max train score {}% and k={}'.format(max_train_score*100,list(map(lambda x: x+1, train_score_i))))

# max train score is 100% at k=1
#now checking for the test score.
max_test_score=max(test_score)
test_score_i=[i for i,v in enumerate(test_score) if v==max_test_score]
print("max test score{}% and k={}".format(max_test_score*100,list(map(lambda x:x+1,test_score_i))))
# max test score is 79.06% at k=1
# now plotting the figure for each k we tested.
plt.figure(figsize=(10,5))
sn.lineplot(range(1,11),train_score,marker="*",label="train_score")
sn.lineplot(range(1,11),test_score,marker="+",label="test_score")
#it is also giving that with the increasing value of k the train and test accuracy is decreasing.

## we can check the better k value for another way that is elbow method.
error_rate=[]
for i in range(1,20):
    knn1=KNC(i)
    knn1.fit(x_train,y_train)
    pred=knn1.predict(x_test)
    error_rate.append(np.mean(pred != y_test))
plt.figure(figsize=(15,8))
sn.lineplot(range(1,20),error_rate,marker="o",label="k vs error_rate")

# by this we can anayse that by increasing in the k value the error rate is getting high.

## now lets normalise the data and see whether we can improve the accuracy or not.
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_minmax = min_max_scaler.fit_transform(x)
x_norm = pd.DataFrame(x_minmax)

# now again transforming the test and train of the normalised data.

x_norm_train,x_norm_test,y_train,y_test=train_test_split(x_norm,y,test_size=0.2,random_state=5,stratify=y)
train1_score=[]
test1_score=[]

# applying knn algo for different values of k ranges from 1 to 10.
for i in range(1,30):
    knn=KNC(i)
    knn.fit(x_norm_train,y_train)
    train1_score.append(knn.score(x_norm_train,y_train))
    test1_score.append(knn.score(x_norm_test,y_test))
## accessing the model.
#and analysing at  which k value the model is giving highest train sscore and test score.
max_train1_score=max(train1_score)
train1_score_i= [i for i, v in enumerate(train1_score) if v == max_train1_score]
print('max train score {}% and k={}'.format(max_train1_score*100,list(map(lambda x: x+1, train1_score_i))))

# max train score is 100% at k=1
#now checking for the test score.
max_test1_score=max(test1_score)
test1_score_i=[i for i,v in enumerate(test1_score) if v==max_test1_score]
print("max test score{}% and k={}".format(max_test1_score*100,list(map(lambda x:x+1,test1_score_i))))
# max test score is 81.39% at k=1
#by normalizing our data we were able to improve the accuracy by 2%.
# now plotting the figure for each k we tested.
plt.figure(figsize=(10,5))
sn.lineplot(range(1,30),train1_score,marker="*",label="train_score")
sn.lineplot(range(1,30),test1_score,marker="+",label="test_score")
#it is also giving that with the increasing value of k the train and test accuracy is decreasing.

