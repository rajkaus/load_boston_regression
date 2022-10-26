#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import library 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_boston # import dataset


# In[85]:


boston=load_boston() #loading boston dataset


# In[86]:


boston.keys()


# In[ ]:


print(boston.DESCR)


# In[ ]:


print(boston.data) #input feature data


# In[ ]:


print(boston.target) #output data


# In[ ]:


print(boston.feature_names)


# In[5]:


#preparing the dataframe by passing in dataframe(data,column)
dataset=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[ ]:


dataset


# In[10]:


dataset['Price']= boston.target # adding price feature to whole dataset


# In[ ]:


dataset.head()  #price added in dataset


# In[ ]:


dataset.info()  #check datatype of independent feature, if object or categorical then need to change to numerical 


# In[ ]:


dataset.describe() #gives all information


# In[ ]:


#check missing value
dataset.isnull().sum()


# In[ ]:


dataset.corr() # checking correlation to gain some info from data, if value is more than 95 %, it means, we can use any one feature for model taining


# In[ ]:


sns.pairplot(dataset)


# In[ ]:


sns.set(rc={'figure.figsize':(10,8)}) # setting figure size
sns.heatmap(dataset.corr(),annot=True)


# In[ ]:


#price is dependent feature
plt.scatter(dataset['CRIM'],dataset['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")


# In[ ]:


plt.scatter(dataset['RM'],dataset['Price']) #checking correlation usin scatter


# In[ ]:


sns.regplot(x='RM',y='Price',data=dataset) # plot data and linerar regression fit
#shaded region meand lasso and ridge, based on lambda changes
#wherver more data points are there, less movement happened and whever less data ponits were there, more movements happened


# In[ ]:


sns.regplot(x='CRIM',y='Price',data=dataset) # plot data and linerar regression fit


# In[ ]:


sns.regplot(x='LSTAT',y='Price',data=dataset) # plot data and linerar regression fit


# In[ ]:


sns.boxplot(dataset['Price'])  # no need to check, output feature


# In[ ]:


sns.boxplot(dataset['CRIM'])


# In[11]:


#independent and dependent features
X=dataset.iloc[:,:-1]


# In[12]:


X


# In[13]:


y=dataset.iloc[:,-1]


# In[14]:


y   #series for deendent


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.33,random_state=10)


# In[17]:


X_train


# In[18]:


y_train #output of X_train


# In[19]:


#size of X_train = size of y_train(rows)
X_train.shape


# In[20]:


y_train.shape


# In[21]:


X_test.shape,y_test.shape


# In[22]:


#feature engineering
#with feature scaling, we can reach global minima qucikly


# In[26]:


#standardization or feature scaling the datasets
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[27]:


X_train_scaler=scaler.fit_transform(X_train) #z-score , change the data


# In[28]:


X_test_scaler=scaler.transform(X_test) # to avoid data leakage, model should not know about anything about test data(here, in test data z score will use training meand and SD to compute)


# # Model Training

# In[29]:


from sklearn.linear_model import LinearRegression


# In[30]:


regression=LinearRegression()


# In[31]:


regression


# In[33]:


regression.fit(X_train_scaler,y_train) #find coefficient, independent feature and y_train dependent feature


# In[34]:


#print the coefficient
print(regression.coef_)  # 13 as we 13 feature


# In[35]:


print(regression.intercept_)


# In[36]:


#prediction for test data
reg_pred=regression.predict(X_test_scaler)


# In[37]:


reg_pred


# In[ ]:


##assumption for test data


# In[38]:


plt.scatter(y_test,reg_pred) # wrt to test data and pred data, linear should come
plt.xlabel("Test truth data")
plt.ylabel("Test predicted data")


# In[40]:


##residual(actual-predicted)
residuals=y_test-reg_pred


# In[41]:


residuals


# In[42]:


sns.displot(residuals,kind="kde")  #it should come as normal distribution for residuals


# In[ ]:


#skewed in right hand side,  because of outliers


# In[ ]:


##scatter plot with prediction and residuals
##uniform distribution
plt.scatter(reg_pred,residuals) # no shape


# In[43]:


##performance metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[44]:


print(mean_squared_error(y_test,reg_pred))
print(mean_absolute_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# In[45]:


# R square and adjusted R sqaure
from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)


# In[46]:


score


# In[49]:


##adusted_r_sqaure
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test_scaler.shape[1]-1)


# In[52]:


## ridge
from sklearn.linear_model import Ridge


# In[54]:


ridge=Ridge()


# In[78]:


rid=ridge.fit(X_train_scaler,y_train)


# In[79]:


rid_pred=rid.predict(X_test_scaler)


# In[80]:


##residual(actual-predicted)
residuals=y_test-rid_pred


# In[81]:


residuals


# In[82]:


print(mean_squared_error(y_test,rid_pred))
print(mean_absolute_error(y_test,rid_pred))
print(np.sqrt(mean_squared_error(y_test,rid_pred)))


# In[83]:


# R square and adjusted R sqaure
from sklearn.metrics import r2_score
score_ridge=r2_score(y_test,rid_pred)


# In[84]:


score_ridge


# In[ ]:





# In[ ]:




