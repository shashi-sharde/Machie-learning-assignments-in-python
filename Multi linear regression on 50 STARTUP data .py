# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:02:20 2020

@author: shashi
"""

#Q.1. Prepare a prediction model for profit of 50_startups data.
#Do transformations for getting better predictions of profit and make a table containing R^2 value for each prepared model.
#R&D Spend -- Research and devolop spend in the past few years
#Administration -- spend on administration in the past few years
#Marketing Spend -- spend on Marketing in the past few years
#State -- states from which data is collected
#Profit  -- profit of each state in the past few years
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sn
startup=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on multilinear regression\\50_Startups.csv")
startup
startup.corr()

startup.columns
startup.columns=["r","admin","ms","state","Prof"]
startup.columns
startup
sn.pairplot(startup)

x=startup.iloc[:,:3]
x
y=startup.iloc[:,4:]
y
mod1=smf.ols("y~x",data=startup).fit() ##r2 - 0.95
mod1
mod1.params
mod1.summary()

rsq_mod1=smf.ols("y~x",data=startup).fit().rsquared
rsq_mod1



rsq_r=smf.ols("r~ms+admin",data=startup).fit().rsquared
vif_r=1/(1-rsq_r)
vif_r

rsq_ms=smf.ols("ms~admin+r",data=startup).fit().rsquared
vif_ms=1/(1-rsq_ms)
vif_ms

rsq_admin=smf.ols("admin~ms+r",data=startup).fit().rsquared
vif_admin=1/(1-rsq_admin)
vif_admin


print(mod1.conf_int(0.05))

pred1=mod1.predict(x)
pred1


plt.scatter(startup.Prof,pred1,c="r");plt.xlabel("observed");plt.ylabel("fitted")
import statsmodels.api as sm
sm.graphics.influence_plot(mod1)

startup=startup.drop(startup.index[[48,49]],axis=0)
startup

mod1_new=smf.ols("Prof~r+ms+admin",data=startup).fit()
mod1_new.params
mod1_new.summary()

rsq_mod1_new=smf.ols("Prof~r+ms+admin",data=startup).fit().rsquared
rsq_mod1_new
x1=startup.iloc[:,:3]
x1
pred1_new=mod1_new.predict(x1)
pred1_new
pred1_new=pred1_new

plt.scatter(startup.Prof,pred1_new,c="g");plt.xlable("observed");plt.ylabel("fitted")

## transforamation of variable.

mod2=smf.ols("Prof~np.log(r+ms+admin)",data=startup).fit()
mod2.params
mod2.summary()

rsq_mod2=smf.ols("Prof~np.log(r+ms+admin)",data=startup).fit().rsquared
rsq_mod2 ## r^2=0.665

## Again transforminig the variable

mod3=smf.ols("Prof~np.sqrt(r+ms+admin)",data=startup).fit()
mod3.params
mod3.summary()

rsq_mod3=smf.ols("Prof~np.sqrt(r+ms+admin)",data=startup).fit().rsquared
rsq_mod3 ## R^2=0.71

## again transforming the variables
mod4=smf.ols("np.sqrt(Prof)~r+ms+admin",data=startup).fit()
mod4.params
mod4.summary()

rsq_mod4=smf.ols("np.sqrt(Prof)~r+ms+admin",data=startup).fit().rsquared
rsq_mod4 ## R^2=0.956

## again transforming the variables
mod5=smf.ols("np.log(Prof)~r+ms+admin",data=startup).fit()
mod5.summary()

rsq_mod5=smf.ols("np.log(Prof)~r+ms+admin",data=startup).fit().rsquared
rsq_mod5 ##R^2 -0.92
## table containing R^squared values for various values.


D={"MODELS":["mod1","mod2","mod3","mod4","mod5"],"R_squared":[rsq_mod1_new,rsq_mod2,rsq_mod3,rsq_mod4,rsq_mod5]}
R_squared=pd.DataFrame(D)
R_squared



plt.scatter(startup.Prof,pred1_new)
## now by analysing this the model 1 is having the best R_squared value .so the model 1 will be our best fit model.

##  VARIABLES R&D SPEND ,ADMINISTRATION AND MONEY SPENDS ARE POSITIVELY CORELATED TO EACH OTHER.
## VARIABLE R&D EFFECTING MORE ON THE DEPENDENT VARIABLE PROFIT WITH CORELATED VALUE 0.97
##  BY ANLYSING THE PLOT OF INPUT AND OUTPUT WE CAN SAY THAT THE MODEL IS PREDICTING NEARLY TOWARDS THE ACTUAL VALUE BECAOUSE IT IS FIINDING APPROX A STRAIGHT LINE.