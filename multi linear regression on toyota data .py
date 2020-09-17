# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:44:39 2020

@author: shashi
"""
##Consider only the below columns and prepare a prediction model for predicting Price.

#Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

 




import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
import seaborn as sn

toyota=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on multilinear regression\\ToyotaCorolla.csv",engine="python")
toyota

t1=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on multilinear regression\\ToyotaCorolla.csv",encoding= 'unicode_escape')
t1

toyota.corr()
toyota.isnull().sum()

toyota.drop(toyota.columns[[0,1,4,5,7,9,10,11,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]],axis=1,inplace=True)
toyota

toyota.corr()

print(toyota.corr())
sn.pairplot(toyota)


print(toyota.columns)
x=toyota.iloc[:,1:9]
print(x)
y=toyota.iloc[:,0]
print(y)

## creating model
mod1=smf.ols("Price~KM+HP+cc+Quarterly_Tax+Age_08_04+Doors+Weight+Gears",data=toyota).fit()
print(mod1.summary())
rsq_mod1=smf.ols("Price~KM+HP+cc+Quarterly_Tax+Age_08_04+Doors+Weight+Gears",data=toyota).fit().rsquared
rsq_mod1 ## r^2 value is 0.86
pred1=mod1.predict(x)
print(pred1)
plt.scatter(toyota.Price,pred1,c="g");plt.xlable("actual");plt.ylabel("fitted")

## in the build model the independent variable cc and doors are having the p value greater than 0.05
# so checkin individualy how they are making the significant impact on the price or not.
mod1_cc=smf.ols("Price~cc",data=toyota).fit()
mod1_cc.summary()

mod1_cd=smf.ols("Price~cc+Doors",data=toyota).fit()
mod1_cd.summary()

mod1_doors=smf.ols("Price~Doors",data=toyota).fit()
mod1_doors.summary()

## now checking for the vif values.
rsq_doors=smf.ols("Doors~KM+HP+cc+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
vif_doors=1/(1-rsq_doors)
print(vif_doors)

rsq_KM=smf.ols("KM~Doors+HP+cc+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
vif_KM=1/(1-rsq_KM)
print(vif_KM)

rsq_HP=smf.ols("HP~Doors+KM+cc+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
vif_HP=1/(1-rsq_HP)
print(vif_HP)

rsq_cc=smf.ols("cc~Doors+KM+HP+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
vif_cc=1/(1-rsq_cc)
print(vif_cc)

rsq_Quarterly_Tax=smf.ols("Quarterly_Tax~Doors+KM+cc+HP+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
vif_Quarterly_Tax=1/(1-rsq_Quarterly_Tax)
print(vif_Quarterly_Tax)


rsq_Weight=smf.ols("Weight~Quarterly_Tax+Doors+KM+cc+HP+Age_08_04+Gears",data=toyota).fit().rsquared
vif_Weight=1/(1-rsq_Weight)
print(vif_Weight)

rsq_Gears=smf.ols("Gears~Weight+Quarterly_Tax+Doors+KM+cc+HP+Age_08_04",data=toyota).fit().rsquared
vif_Gears=1/(1-rsq_Gears)
print(vif_Gears)

##  ALL THE VIF VALUES FALLS UNDER LESS THAN 10 SO WE CAN CONSIDER ALL THE  INDEPENDENT VARIABLES.
## here vif values not makin certain significance but by considering all the espect like standard error,coeficients,p_values the variable "DDORS" is not making much significance on the price and also providing the multicolinearity.
##so removing the doors

mod1_new=smf.ols("Price~KM+HP+cc+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit()
print(mod1_new.summary())
rsq_mod1_new=smf.ols("Price~KM+HP+cc+Quarterly_Tax+Age_08_04+Weight+Gears",data=toyota).fit().rsquared
rsq_mod1_new ## r^2 value is 0.86
pred1_new=mod1_new.predict(x)
print(pred1_new)
plt.scatter(toyota.Price,pred1_new,c="g");plt.xlable("actual");plt.ylabel("fitted")

mod1_new.conf_int(0.05)
## now we have improved r^2 value.that is 86%.
