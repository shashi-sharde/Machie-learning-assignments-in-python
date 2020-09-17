#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Required libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 
from pandas_profiling import ProfileReport


# In[2]:


# Reading the dataset.

df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\PlasticSales.csv")
df


# In[3]:


month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
p = df["Month"][1]
p[0:3]


# In[4]:


df['month']=0

for i in range(60):
    p=df['Month'][i]
    df['month'][i]=p[0:3]
    


# In[13]:


#creating dummy variables for the months i have created
dummy=pd.DataFrame(pd.get_dummies(df['month']))

plastic=pd.concat((df,dummy),axis=1)
#creating newcolumn t
t= np.arange(1,61)
#adding the column t and t_squared in the data
plastic['t']=t
plastic['t_square']=plastic['t']*plastic['t']
#performing the log operation on the sales data and then adding in to in the main data .
log_plstc=np.log(plastic['Sales'])
plastic['log_plstc']=log_plstc
plastic


# In[14]:


#EDA
report=ProfileReport(plastic,title="Profile Report of the Plastic data",explorative=True)


# In[15]:


report.to_widgets()


# In[16]:


#REPORT SAYS THAT:
#1) THERE ARE NO MISSING VALUES
#2) NO DUPLICATE ROWS
#3) NO MULTICOLINEARITY


# In[18]:


#CHECKING THE VARIABLE  DISTRIBUTION THROUGHOUT ALL THE TIME PERIOD AVAILABLE IN THE DATA
df.Sales.plot()


# In[ ]:


#the plot seems like upword linear trend with NO  seasonality
#lets check the same with building the different  forcasting models based approaches.


# In[20]:


#splitting the data
train= plastic.head(44)
test=plastic.tail(16)


# In[21]:


#linear model
linear= smf.ols('Sales~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Sales'])-np.array(predlin))**2))
rmselin


# In[22]:


#quadratic model
quad=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmsequad


# In[23]:


#exponential model
expo=smf.ols('log_plstc~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[24]:


#additive seasonality
additive= smf.ols('Sales~Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predadd))**2))
rmseadd


# In[25]:


#additive seasonality with linear trend
addlinear= smf.ols('Sales~t+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
predaddlinear

rmseaddlinear=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear


# In[26]:


#additive seasonality with quadratic trend
addquad=smf.ols('Sales~t+t_square+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad


# In[27]:


#multiplicative seasonality
mulsea=smf.ols('log_plstc~Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmsemul= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul


# In[28]:


#multiplicative seasonality with linear trend
mullin= smf.ols('log_plstc~t+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin


# In[29]:


#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_plstc~t+t_square+Apr+Aug+Dec+Feb+Jan+Jul+Jun+Mar+May+Nov+Oct+Sep ',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Apr','Aug','Dec','Feb','Jan','Jul','Jun','Mar','May','Nov','Oct','Sep']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad


# In[30]:


#tabulating the rmse values

data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse


# In[ ]:


# the  model with  linear trend with no seasonality is haing the least rmse value. so i will be selecting that model


# In[31]:


#final model with least rmse value

final= smf.ols('Sales~t',data=plastic).fit()
pred= pd.Series(final.predict(plastic))
pred


# In[32]:


#Reading the initial data.
data=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\PlasticSales.csv")

#Adding column with actual predicted value.
data["PREDICTED SALES"]=pred
data


# In[33]:


#the predicted values are floating point no so converting them in to round figure.
data['PREDICTED SALES']=data['PREDICTED SALES'].apply(np.ceil)
data

