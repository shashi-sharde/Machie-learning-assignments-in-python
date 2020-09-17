# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 08:17:48 2020

@author: shashi
"""

## assignment on simple linear regression
## Q.1) Calories_consumed-> predict weight gained using calories consumed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
consumed=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on simple linear regression\\calories_consumed.csv")
consumed
consumed.columns
consumed.columns=["gained","cal_consumed"]
consumed.columns
plt.hist(consumed.gained)
plt.boxplot(consumed.gained)
plt.hist(consumed.cal_consumed)
plt.boxplot(consumed.cal_consumed)
plt.plot(consumed.gained,consumed.cal_consumed,"r+");
plt.xlabel('calConsumed')
plt.ylabel("gained")
consumed.cal_consumed.corr(consumed.gained)
x=consumed.iloc[:,1:]
x
y=consumed.iloc[:,:-1]
y
import statsmodels.formula.api as smf
model1=smf.ols("y~x",data=consumed).fit()##r2-0.888
model1.params
model1.summary()
print(model1.conf_int(0.05))
y_pred=model1.predict(x)
y_pred
plt.scatter(x=consumed["cal_consumed"],y=consumed["gained"],color="red");
plt.plot(consumed["cal_consumed"],y_pred,color="black");
plt.xlabel("CAL_CONSUMED");
plt.ylabel("GAINED")

## now transforming the variables to get better r2 value
model2=smf.ols("y~np.log(x)",data=consumed).fit()##r2-0.792
model2.params
model2.summary()
y_pred1=model2.predict(np.log(x))
y_pred1
plt.scatter(x=consumed["cal_consumed"],y=consumed["gained"],color="blue");
plt.plot(consumed["cal_consumed"],y_pred1,color="red");
plt.xlabel("CAL_CONSUMED");
plt.ylabel("GAINED");

## r2 value is get decreased in order to comparing with model1
## again transforming
model3=smf.ols("y~np.sqrt(x)",data=consumed).fit()##r2-0.845
model3.summary()
y_pred2=model3.predict(np.sqrt(x))
y_pred2
plt.scatter(x=consumed["cal_consumed"],y=consumed["gained"],color="red");
plt.plot(consumed["cal_consumed"],y_pred2,color="green");
plt.xlabel("CAL_CONSUMED");
plt.ylabel("GAINED");

##again its r2 value is less than model 1
model4=smf.ols("np.sqrt(y)~np.sqrt(x)",data=consumed).fit()##r2-0.879
model4.summary()
y_pred3=model4.predict(np.sqrt(x))
y_pred3
plt.scatter(x=consumed["cal_consumed"],y=consumed["gained"],color="green");
plt.plot(consumed["cal_consumed"],y_pred3,color="black");
plt.xlabel("cal_consumed");
plt.ylabel("gained");


## again its r2 value is less than model 1
model5=smf.ols("np.log(y)~x",data=consumed).fit()## r2-0.86
model5.summary()
y_pred4=model5.predict(x)
y_pred4
plt.scatter(x=consumed["cal_consumed"],y=consumed["gained"],color="yellow");
plt.plot(consumed["cal_consumed"],y_pred4,color="blue");
plt.xlabel("CAL_CONSUMED");
plt.ylabel('gained');

##By analysing all the above models model 1 is having the best r2 value.
## so model 1 is the best fit model.

##    Q.2. Delivery_time -> Predict delivery time using sorting time


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

Time=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on simple linear regression\\delivery_time.csv")
Time
Time.columns
Time.columns=["DT","ST"]
Time.columns
plt.hist(Time.ST)
plt.hist(Time.DT)
plt.boxplot(Time.ST)
plt.boxplot(Time.DT)
plt.plot(Time.ST,Time.DT,"r*");plt.xlabel("SORTING TIME");plt.ylabel("DELIEVERY TIME")
Time.DT.corr(Time.ST)##correlation=0.82599
##model building.
mod1=smf.ols("DT~ST",data=Time).fit()##r2-0.666
mod1.params
mod1.summary()
pred1=mod1.predict(Time.ST)
pred1
pred1.corr(Time.ST)

print(mod1.conf_int(0.05))

plt.scatter(Time.ST,Time.DT,c="b");plt.plot(Time.ST,pred1,c="black");plt.xlabel("SORTED TIME");plt.ylabel("DELIEVERY TIME");
pred1.corr(Time.DT)
## transforing variables for accurecy
mod2=smf.ols("DT~np.log(ST)",data=Time).fit()##r2-0.679
mod2.params
mod2.summary()
pred2=mod2.predict(Time.ST)
pred2
pred2.corr(Time.ST)

print(mod2.conf_int(0.05))

plt.scatter(x=Time["ST"],y=Time["DT"],color="blue");plt.plot(Time["ST"],pred2,color="black");plt.xlabel("SORTED TIME");plt.ylabel("DELievery time")

# Exponential transformation
mod3=smf.ols("np.log(DT)~ST",data=Time).fit() ##r2=0.696
mod3.summary()
pred=mod3.predict(Time.ST)
pred
pred3=np.exp(pred)
pred3
pred3.corr(Time.ST)

print(mod3.conf_int(0.05))

plt.scatter(x=Time["ST"],y=Time["DT"],color="black");plt.plot(Time["ST"],pred3,color="red");plt.xlabel("SORTING TIME");plt.ylabel("DELIEVERY TIME")
##PREDICTED VS ACTUAL PLOT.
plt.scatter(x=pred3,y=Time.DT);plt.xlabel("PREDCITED");plt.ylabel("actual")

##quadratic model
Time["ST_sq"]=Time.ST*Time.ST 
mod_quad=smf.ols("np.log(DT)~ST+ST_sq",data=Time).fit()## r2-0.739
mod_quad.params
mod_quad.summary()
pred_quad=mod_quad.predict(Time)
pred_quad
pred4=np.exp(pred_quad)
pred4
pred4.corr(Time.ST)

print(mod_quad.conf_int(0.05))

plt.scatter(Time.ST,Time.DT,c="b");plt.plot(Time.ST,pred4,c="r")


##transforming the variables
mod_5=smf.ols("DT~np.sqrt(ST)",data=Time).fit()## r2-0.68
mod_5.params
mod_5.summary()
pred5=mod_5.predict(Time.ST)
pred5
pred5.corr(Time.ST)

print(mod_5.conf_int(0.05))

plt.scatter(Time.ST,Time.DT,c="r");plt.plot(Time.ST,pred5,c="b");plt.xlabel("sorting time");plt.ylabel("Delievery time")
## again transforming the variables,
mod_6=smf.ols("np.sqrt(DT)~ST",data=Time).fit()##r2-0.688
mod_6.params
mod_6.summary()
pred6=mod_6.predict(Time.ST)
pred6
pred6.corr(Time.ST)

print(mod_6.conf_int(0.05))

plt.scatter(Time.ST,Time.DT,c="r");plt.plot(Time.ST,pred6,c="b");plt.xlabel("SORTING TIME");plt.ylabel("Delievery time")
 ## now by analysing vairous  created models the  quadratic model is having the best R_squared value that will be our best fit model.
 
 
 
 
 ## Q.3) Emp_data -> Build a prediction model for Churn_out_rate
 
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

churn=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on simple linear regression\\emp_data.csv")
churn
churn.columns=["salery","cor"]
churn.columns
plt.hist(churn.salery)
plt.plot(churn.salery)
plt.hist(churn.cor)
plt.plot(churn.cor)
plt.plot(churn.salery,churn.cor,"r+");plt.xlabel("salery hike");plt.ylabel("Churn out rate")

## model building
model1=smf.ols("cor~salery",data=churn).fit()## r2-0.81
model1.params
model1.summary()
pred1=model1.predict(churn.salery)
pred1
pred1.corr(churn.salery)

print(model1.conf_int(0.05))

plt.scatter(churn.salery,churn.cor,c="b");plt.plot(churn.salery,pred1,c="r");plt.xlabel("salery hike ");plt.ylabel("churn out rate")

## transforming variBLES 
model2=smf.ols("cor~np.log(salery)",data=churn).fit()## r2-0.83
model2.params
model2.summary()
pred2=model2.predict(churn.salery)
pred2
pred2.corr(churn.salery)

print(model2.conf_int(0.05))

plt.scatter(churn.salery,churn.cor,c="g");plt.plot(churn.salery,pred2,c="black");plt.xlabel("salery");plt.ylabel("churn out rate")
## exponential transformation
model3=smf.ols("np.log(cor)~salery",data=churn).fit() ##r2-0.858
model3.params
model3.summary()
pred3=model3.predict(churn.salery)
pred3
pred_exp=np.exp(pred3)
pred_exp
pred_exp.corr(churn.salery)

print(model3.conf_int(0.05))

plt.scatter(churn.salery,churn.cor,c="g");plt.plot(churn.salery,pred_exp,c="black");plt.xlable("salery");plt.ylabel("churn out rate")

## quadratic model
churn["salery_sq"]=churn.salery*churn.salery
model_quad=smf.ols("np.log(cor)~salery+salery_sq",data=churn).fit()##r2-97.9
model_quad.params
model_quad.summary()
pred4=model_quad.predict(churn)
pred4
pred_quad=np.exp(pred4)
pred_quad

print(model_quad.conf_int(0.05))

plt.scatter(churn.salery,churn.cor,c="g");plt.plot(churn.salery,pred_quad,c="black");plt.xlabel("SALERY HIKE");plt.ylabel("CHURN_OUT_RATE")
pred_quad.corr(churn.salery)

## by analysing the r_squared value of each model we conclude that quadratic model is having high r_squared value.
# so Quadrativ model will be the our best fit model.


#4) Salary_hike -> Build a prediction model for Salary_hike


#Do the necessary transformations for input variables for getting better R^2 value for the model prepared.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

hike=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on simple linear regression\\Salary_Data.csv")
hike
hike.columns=["years","sal_hike"]
hike.columns
plt.plot(hike.years,hike.sal_hike,"r*");plt.xlabel("year of experience");plt.ylabel("salery hike")

(hike.sal_hike).corr(hike.years)
## model building
model1=smf.ols("sal_hike~years",data=hike).fit() ##r2-90.955
model1.params
model1.summary()

pred1=model1.predict(hike.years)
pred1

pred1.corr(hike.years)

print(model1.conf_int(0.05))

plt.scatter(hike.years,hike.sal_hike,c="g");plt.plot(hike.years,pred1,c="black");plt.xlabel("experience of years");plt.ylabel("Salery hike")

## transformation of variables.
 
model2=smf.ols("sal_hike~np.log(years)",data=hike).fit() ##r2-0.84
model2.params
model2.summary()

pred2=model2.predict(hike.years)
pred2
pred2.corr(hike.years)

print(model2.conf_int(0.05))

plt.scatter(hike.years,hike.sal_hike,c="g");plt.plot(hike.years,pred2,c="black");plt.xlabel("YEARS OF EXPERIENCE");plt.ylabel("SALERY HIKE")

## exponetial transformation
model3=smf.ols("np.log(sal_hike)~years",data=hike).fit()##r2-0.93
model3.params
model3.summary()

pred3=model3.predict(hike.years)
pred3
pred_exp=np.exp(pred3)
pred_exp

print(model3.conf_int(0.05))

plt.scatter(hike.years,hike.sal_hike,c="g");plt.plot(hike.years,pred_exp,c="black");plt.xlabel("years of experience");plt.ylable("SALERY HIKE")
pred_exp.corr(hike.years)



## quadratic transformation
hike["years_sq"]=hike.years*hike.years
model_quad=smf.ols("np.log(sal_hike)~years+years_sq",data=hike).fit()##r2-0.945
model_quad.params
model_quad.summary()
pred4=model_quad.predict(hike)
pred4
pred_quad=np.exp(pred4)
pred_quad

pred_quad.corr(hike.years)


plt.scatter(hike.years,hike.sal_hike,c="g");plt.plot(hike.years,pred_quad,c="black");plt.xlabel("YEARS OF EXPERIENCE");plt.ylabel("SALERY_HIKE")

print(model_quad.conf_int(0.05))
## now by analysing the r_square value of each model the model1 is having the highest r_squared value.
#so the model 1 will be our best fit model.