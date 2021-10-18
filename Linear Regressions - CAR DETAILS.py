#!/usr/bin/env python
# coding: utf-8

# In[9]:


import warnings
warnings.simplefilter("ignore")


# In[10]:


import pandas as pd
import numpy as np


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


dataset = pd.read_csv('Car details.csv')
dataset


# In[63]:


dataset.shape


# In[64]:


dataset.head()


# In[65]:


dataset=dataset.loc[::,['km_driven','year','fuel','owner','mileage','engine','selling_price']]


# In[66]:


dataset


# In[67]:


x = dataset.iloc[:,0]


# In[68]:


x


# In[69]:


x.shape


# In[70]:


x = dataset.iloc[:,0].values.reshape(-1,1)


# In[71]:


x.shape


# In[72]:


y = dataset.iloc[:,-1].values.reshape(-1,1)


# In[73]:


y.shape


# In[74]:


y


# In[76]:


plt.scatter(x,y)
plt.show


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[79]:


x_train.shape


# In[80]:


x_test.shape


# In[81]:


y_train.shape


# In[82]:


y_test.shape


# In[83]:


from sklearn.linear_model import LinearRegression


# In[84]:


lm = LinearRegression()


# In[85]:


lm.fit(x_train,y_train)


# In[86]:


y_pred = lm.predict(x_test)


# In[87]:


y_pred


# In[90]:


plt.scatter(x,y,color='blue')
plt.plot(x_test,y_pred,color='red')

