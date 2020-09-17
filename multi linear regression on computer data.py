# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:52:36 2020

@author: shashi
"""
#Predict Price of the computer

#A dataframe containing :

#price : price in US dollars of 486 PCs

#speed : clock speed in MHz

#hd : size of hard drive in MB

#ram : size of Ram in in MB

#screen : size of screen in inches

#cd : is a CD-ROM present ?

#multi : is a multimedia kit (speakers, sound card) included ?

#premium : is the manufacturer was a "premium" firm (IBM, COMPAQ) ?

#ads : number of 486 price listings for each month

#trend : time trend indicating month starting from January of 1993 to November of 1995.




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sn 
 
computer=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on multilinear regression\\Computer_Data.csv")
computer
computer.drop(computer.columns[[0]],axis=1,inplace=True)
computer







plt.scatter(computer.hd,computer.price,c="g")
computer.hd.corr(computer.price)
plt.scatter(computer.speed,computer.price,c="r")
## by analysing the scatter plot we are not be able to get any info out of it.so will go for the bar graph for each and every independent variable 
# to see all the impact of the variable on the price of the computer.
 
## analysing the effect of all the independent variable on the price.

##  first mapping the unique int  values of ram in  object.
computer['ram'] = computer['ram'].map({2: '2', 24: '24', 32: '32', 4: '4', 8: '8', 16: '16'})

ramgroup = computer.groupby('ram').mean()
ram = [ram for ram, computer in computer.groupby('ram')]

plt.barh(ram, ramgroup['price'], color='tomato')
plt.title('The Price of Computers By Quantity of RAMs', color='tomato')
plt.xlabel('Price', color='tomato')
plt.ylabel('RAM')

## as we can see the price of computer is getting high by incresing the quantity of ram
## now analysing for the cd
cdgroup = computer.groupby('cd').mean()
cd = [cd for cd, computer in computer.groupby('cd')]

plt.barh(cd, cdgroup['price'], color='green')
plt.title('The Price of Computers with cd', color='green')
plt.xlabel('Price', color='green')
plt.ylabel('cd')

# so by graph it clear that with cd the price of computer is getting high.

# now analysing the price by hard disk space.
hdgroup = computer.groupby('hd').mean()
hd = [hd for hd, computer in computer.groupby('hd')]

sn.barplot(hd, hdgroup['price'], label='Price', color='forestgreen', alpha=0.9)
plt.legend()
plt.title("Computer's Price by Hard-Desk Space")
plt.xlabel('Hard-Desk space')
plt.show()


#we are not be able to interpret from the graph because it is having too much of values so mapping the values.

computer.hd.describe()
## it is having min value 80 and max value 2100 so mapping all the values in between.

computer.loc[computer['hd'] <= 300, 'hd'] = 300
computer.loc[(computer['hd'] > 300) & (computer['hd'] <= 600), 'hd'] = 600
computer.loc[(computer['hd'] > 600) & (computer['hd'] <= 900), 'hd'] = 900
computer.loc[(computer['hd'] > 900) & (computer['hd'] <= 1200), 'hd'] = 1200
computer.loc[(computer['hd'] > 1200) & (computer['hd'] <= 1500), 'hd'] = 1500
computer.loc[(computer['hd'] > 1500) & (computer['hd'] <= 1800), 'hd'] = 1800
computer.loc[(computer['hd'] > 1800) & (computer['hd'] <= 2100), 'hd'] = 2100

hdgroup = computer.groupby('hd').mean()
hd = [hd for hd, computer in computer.groupby('hd')]

sn.barplot(hd, hdgroup['price'], label='Price', color='forestgreen', alpha=0.9)
plt.legend()
plt.title("Computer's Price by Hard-Desk Space")
plt.xlabel('Hard-Desk space')
plt.show()
#now with this we can analyse the impact of hard disk spce on the price.more the hard disk high the price.but somestage between 1500 to 1800 disk space the price are geting low.

# now cheking price with or without preimium'

pregroup = computer.groupby('premium').mean()
pre = [premium for premium, computer in computer.groupby('premium')]

sn.barplot(pre, pregroup['price'], label='Price', color='darkgreen', alpha=0.9)
plt.legend()
plt.show()
plt.xlabel('premium')

# price is getting high with premium

##PREPROCESSING OF THE DATA

#mapping the values of yes or no in to o and 1
mapping={"yes":1,"no":0}

for col in computer:
    if computer[col].dtypes == object:
        computer[col]=computer[col].map(mapping)
## removing th na values if any
computer.dropna(axis=1, inplace=True)

##viewing the data whether our operation has worked or not
computer    
# devinding the data in to x or y
x=computer.iloc[:,1:9]
x

y=computer.iloc[:,0]
y
    ## other way of doing this is "
   # y1=computer.price
    #y1
    #x1=computer.drop(["price"],axis=1)
    #x1


#creating model
model1=smf.ols("price~speed +hd + screen +cd  +multi+  premium + ads + trend",data=computer).fit()
model1.summary()
## R^2=0.65
pred1=model1.predict(x)
pred1
#BY ANALYSING TH CORRELATIION VALUE AND P VALUE THERE IS NO COLINEARITY BETWEEN THE VARIABLES BECOUSE ALL OF ARE HAVING THE P VALUES LESS THAN 0.05
#transforming the variables
model2=smf.ols("price~np.log(speed +hd + screen +cd  +multi+  premium + ads + trend)",data=computer).fit()
model2.summary()
## r2=0.204
#transforming the variables
model3=smf.ols("np.log(price)~ speed  +  hd + ram + screen + cd + multi + premium +ads + trend",data=computer).fit()
model3.summary()
## R^2=0.783
#transforming the variables
model4=smf.ols("price~np.sqrt(  speed  +  hd + ram + screen + cd + multi + premium + ads + trend)",data=computer).fit()
model4.summary()
##r^2=0.221
#transforming the variables
model5=smf.ols("np.sqrt(price)~speed  +  hd + ram + screen + cd + multi + premium + ads + trend",data=computer).fit()
model5.summary()
#r^2=0.785
PRED5=model5.predict(x)
PRED5
#BY ANALYSING ALL OF THE MODELS MODEL5 IS HAVING BEST R^2 VALUE.SO THATS OUR BEST FIT MODEL.



