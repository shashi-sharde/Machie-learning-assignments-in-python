#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Required libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 
from pandas_profiling import ProfileReport


# In[2]:


#Reading the dataset.

df=pd.read_csv( "C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\Airlines+Data.csv")
df


# In[3]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
p = df["Month"][1]
p[0:3]


# In[4]:


df['month']=0

for i in range(96):
    p=df['Month'][i]
    df['month'][i]=p[0:3]
    


# In[6]:


#creating dummy variables for the months i have created
dummy=pd.DataFrame(pd.get_dummies(df['month']))

air=pd.concat((df,dummy),axis=1)
#creating newcolumn t
t= np.arange(1,97)
#adding the column t and t_squared in the data
air['t']=t
air['t_square']=air['t']*air['t']
#performing the log operation on the Passenger data and then adding in to in the main data .
log_pass=np.log(air['Passengers'])
air['log_pass']=log_pass


# In[7]:


air


# In[8]:


#EDA
report=ProfileReport(air,title="Profile Report of the Airlines data",explorative=True)


# In[29]:


report.to_widgets()


# In[ ]:


#REPORT SAYS THAT:
#1) THERE ARE NO MISSING VALUES
#2) NO DUPLICATE ROWS
#3) NO MULTICOLINEARITY


# In[9]:


#CHECKING THE VARIABLE  DISTRIBUTION THROUGHOUT ALL THE TIME PERIOD AVAILABLE IN THE DATA
df.Passengers.plot()


# In[ ]:


#the plot seems like upword linear trend with NO  seasonality
#lets check the same with building the different  forcasting models based approaches.


# In[10]:


#splitting the data
train= air.head(76)
test=air.tail(19)


# In[11]:


#linear model
linear= smf.ols('Passengers~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Passengers'])-np.array(predlin))**2))
rmselin


# In[12]:


#quadratic model
quad=smf.ols('Passengers~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predquad))**2))
rmsequad


# In[13]:


#exponential model
expo=smf.ols('log_pass~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[14]:


#additive seasonality
additive= smf.ols('Passengers~Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predadd))**2))
rmseadd


# In[15]:


#additive seasonality with linear trend
addlinear= smf.ols('Passengers~t+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
predaddlinear

rmseaddlinear=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddlinear))**2))
rmseaddlinear


# In[16]:


#additive seasonality with quadratic trend
addquad=smf.ols('Passengers~t+t_square+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(predaddquad))**2))
rmseaddquad


# In[19]:


#multiplicative seasonality
mulsea=smf.ols('log_pass~Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmsemul= np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmul)))**2))
rmsemul


# In[20]:


#multiplicative seasonality with linear trend
mullin= smf.ols('log_pass~t+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(predmullin)))**2))
rmsemulin


# In[21]:


#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_pass~t+t_square+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep ',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Passengers'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad


# In[22]:


#tabulating the rmse values

data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse


# In[ ]:


#THE MULTIPLICATIVE SEASONALITY WITH LINEAR TREND   MODEL  IS HAVING THE LEAST RMSE VALUE .SO I WILL BE SELECTING THAT MODEL.


# In[25]:



#final model with least rmse value

final= smf.ols('log_pass~t+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=air).fit()
pred= pd.Series(final.predict(air))
pred


# In[27]:


#Reading the initial data.
data=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\Airlines+Data.csv")

#Adding column with actual predicted value.
data["PREDICTED passengers"]=pred
data


# In[28]:


#above predicted values are the logged values so we need to transform it to its normal values.
data['PREDICTED passengers']=np.exp(data['PREDICTED passengers'])
data


# In[29]:


#the predicted values are floating point no so converting them in to round figure.
data['PREDICTED passengers']=data['PREDICTED passengers'].apply(np.ceil)
data


# In[ ]:


#this is the predicted result .

