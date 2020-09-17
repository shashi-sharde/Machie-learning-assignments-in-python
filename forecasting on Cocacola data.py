#!/usr/bin/env python
# coding: utf-8

# MODEL BASED APPROACH

# In[1]:


#MODEL BASED APPROACH


#Required libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf 
from pandas_profiling import ProfileReport


# In[2]:


# Reading the dataset.

df=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\CocaCola_Sales_Rawdata.csv")
df


# In[20]:


#creating 4 new column for the different values for the various quaters of the year.
quarter=['Q1','Q2','Q3','Q4']
n=df['Quarter'][0]
n[0:2]


# In[21]:


df['quarter']=0

for i in range(42):
    n=df['Quarter'][i]
    df['quarter'][i]=n[0:2]
    


# In[22]:


#creating dummy variables for the quarters i have created
dummy=pd.DataFrame(pd.get_dummies(df['quarter']))

coca=pd.concat((df,dummy),axis=1)
#creating newcolumn t
t= np.arange(1,43)
#adding the column t and t_squared in the data
coca['t']=t
coca['t_square']=coca['t']*coca['t']
#performing the log operation on the sales data and then adding in to in the main data .
log_Sales=np.log(coca['Sales'])
coca['log_Sales']=log_Sales


# In[25]:


coca


# In[26]:


#EDA
report=ProfileReport(coca,title="Profile Report of the CocacolaData",explorative=True)


# In[27]:


report.to_widgets()


# In[ ]:


#REPORT SAYS THAT:
#1) THERE ARE NO MISSING VALUES
#2) NO DUPLICATE ROWS
#3) NO MULTICOLINEARITY


# In[28]:


#CHECKING THE VARIABLE SALES DISTRIBUTION THROUGHOUT ALL THE QUATERS AVAILABLE IN THE DATA
df.Sales.plot()


# In[ ]:


# the plot seems like upword linear trend with additive seasonality
#lets check the same with building the different  forcasting models based approaches.


# In[30]:


#splitting the data
train= coca.head(34)
test=coca.tail(8)


# In[31]:


#linear model
linear= smf.ols('Sales~t',data=train).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test['t'])))
rmselin=np.sqrt((np.mean(np.array(test['Sales'])-np.array(predlin))**2))
rmselin


# In[32]:


#quadratic model
quad=smf.ols('Sales~t+t_square',data=train).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test[['t','t_square']])))
rmsequad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predquad))**2))
rmsequad


# In[33]:


#exponential model
expo=smf.ols('log_Sales~t',data=train).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test['t'])))
predexp
rmseexpo=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[34]:


#additive seasonality
additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
predadd
rmseadd=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predadd))**2))
rmseadd


# In[35]:


#additive seasonality with linear trend
addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
predaddlinear

rmseaddlinear=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear


# In[36]:


#additive seasonality with quadratic trend
addquad=smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test[['t','t_square','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(predaddquad))**2))
rmseaddquad


# In[37]:


#multiplicative seasonality
mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul


# In[38]:


#multiplicative seasonality with linear trend
mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin


# In[39]:


#multiplicative seasonality with quadratic trend
mul_quad= smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=train).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test[['t','t_square','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad


# In[40]:


#tabulating the rmse values

data={'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}
data

Rmse=pd.DataFrame(data)
Rmse


# In[ ]:


#THE MODEL WITH MULTIPLICATIVE SEOSANALITY IS HAVING THE LEAST RMSE VALUE .SO I WILL BE SELECTING THAT MODEL.


# In[45]:


#final model with least rmse value

final= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=coca).fit()
pred= pd.Series(final.predict(coca))
actual_pred = np.exp(pred)
actual_pred


# In[47]:


#Reading the initial data.
data=pd.read_csv("C:\\Users\\shashi\\Downloads\\DATA SCIENCE\\data science assignment\\assignment on forecasting\\CocaCola_Sales_Rawdata.csv")

#Adding column with actual predicted value.
data["PREDICTED SALES"]=actual_pred
data

