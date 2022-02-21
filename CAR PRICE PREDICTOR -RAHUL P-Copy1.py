#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[14]:


car_data=pd.read_csv('car data.csv')


# In[15]:


car_data.head(20)


# In[16]:


car_data.isnull().sum()


# In[10]:


car_data.shape


# In[ ]:





# In[12]:


car_data=car_data.dropna()


# In[13]:


car_data.isnull().sum()


# In[17]:


car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)


car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)


car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[18]:


car_data.head(20)


# In[34]:


#USING MULTIPLE LINEAR REGRESSION
from sklearn import linear_model
regr = linear_model.LinearRegression()
x=np.asanyarray(car_data[['Year','Present_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']])
y=np.asanyarray(car_data[['Selling_Price']])



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=2)
regr.fit(x_train,y_train)

from sklearn import metrics
y_predicted=regr.predict(x_test)
error_score = metrics.r2_score(y_test, y_predicted)
print("R squared Error : ", error_score)


# In[40]:


#USING  LASSO REGRESSION
from sklearn.linear_model import Lasso
lasso_regr=Lasso()

lasso_regr.fit(x_train,y_train)
y_lasso_predicted=lasso_regr.predict(x_test)
error_score = metrics.r2_score(y_test, y_lasso_predicted)
print("R squared Error : ", error_score)


# In[45]:


print("coefficients=",lasso_regr.coef_)


# In[ ]:




