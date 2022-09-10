#!/usr/bin/env python
# coding: utf-8

# # Ml-flow for sales prediction 

# # 1. Import the packages

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")
import plotly 
import plotly.graph_objs as go
import chart_studio.plotly as py
#import plotly.plotly as py
import sklearn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[84]:


import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LogisticRegression
from fast_ml.model_development import train_valid_test_split
#import mlflow
#import mlflow.sklearn
import logging
import matplotlib.pyplot as plt


import dvc.api


# # 2. Data Preparation

# # 2.1. Load Dataset

# In[85]:


# Loan Dataset_store 
 
store_data = pd.read_csv(r"C:\Users\Genet Shanko\Pharmaceutical_Sales_Prediction\DVC_Dataset\Store.csv")


# In[86]:


store_data.head()


# In[87]:


def str_to_date(date):
    return datetime.strptime(date, '%Y-%m-%d').date()


# In[88]:


train_data= pd.read_csv(r"C:\Users\Genet Shanko\Pharmaceutical_Sales_Prediction\DVC_Dataset\train.csv",parse_dates = True,index_col = 'Date')


# In[89]:


train_data.head()


# In[90]:


# have a glance on the datasets
print("# of observations & # of features", train_data.shape)
train_data.head()


# In[91]:


print("# of observations & # of features", store_data.shape)
store_data.head()


# # 2.2. Working on Missing Values

# In[92]:


print("train:\n\n", train_data.isnull().sum(),  
     "\n\nstore:\n\n", store_data.isnull().sum())


# ### 2.3. Remove features with high percentages of missing values

# In[33]:


# remove features
store_data_up = store_data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear','Promo2SinceWeek',
                     'Promo2SinceYear', 'PromoInterval'], axis=1)


# ### 2.4. Replace missing values in features with low percentages of missing values

# In[34]:


# CompetitionDistance is distance in meters to the nearest competitor store
# let's first have a look at its distribution
sns.distplot(store_data_up.CompetitionDistance.dropna())
plt.title("Distributin of Store Competition Distance")


# ##### The distribution is right skewed, so we'll replace missing values with the median.

# In[35]:


# replace missing values in CompetitionDistance with median for the store dataset
store_data_up.CompetitionDistance.fillna(store_data_up.CompetitionDistance.median(), inplace=True)


# ## 2.4.Date Extraction

# In[36]:


# extract year, month, day and week of year from "Date"
train_data['Year'] = train_data.index.year
train_data['Month'] = train_data.index.month
train_data['Day'] = train_data.index.day
train_data['WeekOfYear'] = train_data.index.weekofyear
train_data = train_data.reset_index()


# ## 2.5.Joining Tables

# In[37]:


df = pd.merge(train_data, store_data_up, how='left', on='Store')
df.head()


# In[38]:


len(df)


# ## 2.6. Drop Subsets Of Data Where Might Cause Bias

# In[39]:


# where stores are closed, they won't generate sales, so we will remove this part of the dataset
df = df[df.Open != 0]


# In[40]:


# Open isn't a variable anymore, so we'll drop it
df = df.drop('Open', axis=1)


# In[41]:


# see if there's any opened store with zero sales
df[df.Sales == 0]['Store'].sum()


# In[42]:


# see the percentage of open stored with zero sales
df[df.Sales == 0]['Sales'].sum()/df.Sales.sum()


# In[43]:


# remove this part of data to avoid bias
df = df[df.Sales != 0]


# ## 2.7. Feature Engineering

# In[44]:


# see what variable types we have
df.info()


# In[45]:


# see what's in nominal varibles 
set(df.StateHoliday), set(df.StoreType), set(df.Assortment)


# In[46]:


# StateHoliday indicates a state holiday - a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# convert number 0 to string 0
df.loc[df.StateHoliday == 0,'StateHoliday'] = df.loc[df.StateHoliday == 0,'StateHoliday'].astype(str)


# In[47]:


# make a copy in case I mess up anything 
df1 = df.copy()


# #### 2.7.1 Create new variable "AvgeSales"

# ##### create a variable that calculates monthly average sales for each store

# In[48]:


# calculate weekly average sales
sales = df1[['Year','Month','Store','Sales']].groupby(['Year','Month','Store']).mean()
sales = sales.rename(columns={'Sales':'AvgSales'})
sales = sales.reset_index()


# In[49]:


df1['sales_key']=df1['Year'].map(str) + df1['Month'].map(str) + df1['Store'].map(str)
sales['sales_key']=sales['Year'].map(str) + sales['Month'].map(str) + sales['Store'].map(str)


# In[50]:


# drop extra columns
sales = sales.drop(['Year','Month','Store'], axis=1)
# merge
df1 = pd.merge(df1, sales, how='left', on=('sales_key'))


# ##### 2.7.2. Create new variable "AvgeCustomer"

# create a variable that calculates Monthly average number of customers for each store, becuase daily number of customer is another variable to be predicted

# In[51]:


# calculate weekly average
cust = df1[['Year','Month','Store','Customers']].groupby(['Year','Month', 'Store']).mean()
cust = cust.rename(columns={'Customers':'AvgCustomer'})
cust = cust.reset_index()


# In[52]:


df1['cust_key']=df1['Year'].map(str) + df1['Month'].map(str) + df1['Store'].map(str)
cust['cust_key']=cust['Year'].map(str) + cust['Month'].map(str) + cust['Store'].map(str)


# In[53]:


# drop original feature Customers
df1 = df1.drop('Customers', axis=1)# drop extra columns
cust = cust.drop(['Year', 'Month', 'Store'], axis=1)


# In[54]:


# merge
df1 = pd.merge(df1, cust, how="left", on=('cust_key'))


# ##### 2.7.3. Transform Variable "StateHoliday"

# In[55]:


# 0 - not a state holiday; 1- is on a state holiday
df1['StateHoliday'] = df1.StateHoliday.map({'0':0, 'a':1 ,'b' : 1,'c': 1})


# In[56]:


# drop extra columns
df1 = df1.drop(['cust_key','sales_key','Store','Date'], axis=1)


# # 3. Exploratory Data Analysis

# In[57]:


# becasue my computer keeps crashing, I had to sample the dataset 
dfv = df.sample(n=5000, random_state=1)


# In[58]:


# set up credential file for plotly
#py.tools.set_credentials_file(username='mei_zmyang', api_key='Z8Jn8zb2xXh4lfckv9xa')


# ### 3.1. Correlation Heatmap

# In[59]:


corr = df1.corr()


# In[60]:


mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (11, 9))
sns.heatmap(corr, mask = mask,
            square = True, linewidths = .5, ax = ax, cmap = "BuPu")
plt.title("Correlation Heatmap", fontsize=20)


# ### 3.2.Sales Distribution

# In[62]:


sales_dist = [go.Histogram(x=dfv.Sales, marker=dict(color='pink'))]
layout2 = go.Layout(title='Sales Distribution', xaxis=dict(title='daily sales in $'), yaxis=dict(title='number of observations'))
fig2 = go.Figure(data=sales_dist, layout=layout2)
#py.iplot(fig2)


# ### 3.3.Customer Distribution

# In[67]:


cust_dist = [go.Histogram(x=dfv.Customers, marker=dict(color=('blue')))]
layout3 = go.Layout(title='Customer Distribution', 
                   xaxis=dict(title='daily total number of customers'), yaxis=dict(title='number of observations'))
fig3 = go.Figure(data=cust_dist, layout=layout3)
#py.iplot(fig3)


# ### 3.4. Sales Over Time

# In[69]:


store1_2015 = df.query('Store == 1 and Year == 2015')
store1_2013 = df.query('Store == 1 and Year == 2013')
store1_2014 = df.query('Store == 1 and Year == 2014')
trace_2013 = go.Scatter(
                x=store1_2013.Date,
                y=store1_2013.Sales,
                name = "2013",
                opacity = 0.8)

trace_2014 = go.Scatter(
                x=store1_2014.Date,
                y=store1_2014.Sales,
                name = "2014",
                opacity = 0.8)

trace_2015 = go.Scatter(
                x=store1_2015.Date,
                y=store1_2015.Sales,
                name = "2015",
                opacity = 0.8)

data = [trace_2013,trace_2014, trace_2015]
layout = go.Layout(title='Sales Over Time', 
                   xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
fig = go.Figure(data=data, layout=layout)
#py.iplot(fig)


# In[70]:


sns.factorplot(data = dfv, x = 'Month', y = "Sales", 
               col = 'Assortment',
               palette = 'plasma',
               hue = 'StoreType')


# ## 3.5. Sales vs. Competition Distance

# In[75]:


distance_s = [go.Scatter(x=dfv.CompetitionDistance, y=dfv.Sales, mode='markers', 
                     marker=dict(size=20,color=dfv.Customers,
                                 colorbar=dict(title='Number Of Customers'),colorscale='Jet'))]
layout10 = go.Layout(title='Sales vs. Competition Distance', 
                   xaxis=dict(title='Competition Distance'), yaxis=dict(title='Sales'))
fig10 = go.Figure(data=distance_s, layout=layout10)
#py.iplot(fig10)


# ## 3.6.Sales By Promotion

# In[ ]:


s_promo = [go.Box(x=dfv.Promo, y=dfv.Sales,marker=dict(color='purple'), 
                 boxpoints='all', jitter=0.3, pointpos=-1.8)]
layout11 = go.Layout(title='Sales By Promotion', 
                   xaxis=dict(title='Promotion(0-No Promotion; 1-Promotion Period)'), yaxis=dict(title='Total Sales'))
fig11 = go.Figure(data=s_promo, layout=layout11)
py.iplot(fig11)


# In[73]:


promo0 = dfv.query('Promo==0')
promo0 = promo0.rename(columns={'Sales':'No_Promotion_Sales'})
promo1 = dfv.query('Promo==1')
promo1 = promo1.rename(columns={'Sales':'Promotion_Sales'})
x1 = promo0.No_Promotion_Sales
x2 = promo1.Promotion_Sales


# In[74]:


plt.figure(figsize=(10,6))
ax = sns.kdeplot(x1, shade=True, color="r")
ax = sns.kdeplot(x2, shade=True, color="b")
plt.title("Sales By Promotion")
plt.ylabel('Sales')


# # 4.Store Sales Prediction

# In[78]:


dfd = df1.sample(n=50000, random_state=1)


# In[79]:


X = dfd.drop('Sales', axis=1)
y = dfd. Sales


# In[80]:


xd = X.copy()
xd = pd.get_dummies(xd)


# In[97]:


xl = X.copy()

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
xl.StateHoliday = label.fit_transform(xl.StateHoliday)
xl.Assortment = label.fit_transform(xl.Assortment)
xl.StoreType = label.fit_transform(xl.StoreType)


# In[104]:


from packaging.version import parse
import sklearn
if parse(sklearn.__version__) > parse('0.18'):
    from sklearn.model_selection import train_test_split
else:
    from sklearn.cross_validation import train_test_split


# In[105]:



# split training and test datasets
#from sklearn.cross_validation import train_test_split
xd_train,xd_test,yd_train,yd_test = train_test_split(xd,y,test_size=0.3, random_state=1)
xl_train,xl_test,yl_train,yl_test = train_test_split(xl,y,test_size=0.3, random_state=1)


# ## 4.1. inear Regression 

# In[106]:


from sklearn.linear_model import LinearRegression
lin= LinearRegression()
linreg = lin.fit(xd_train, yd_train)


# In[107]:


# definte RMSE function
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(x, y):
    return sqrt(mean_squared_error(x, y))

# definte MAPE function
def mape(x, y): 
    return np.mean(np.abs((x - y) / x)) * 100  
  
# get cross validation scores 
yd_predicted = linreg.predict(xd_train)
yd_test_predicted = linreg.predict(xd_test)

print("Regresion Model Score" , ":" , linreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , linreg.score(xd_test, yd_test))
print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))


# ## 4.2.Bayesian Ridge Regression

# In[108]:


from sklearn.linear_model import BayesianRidge
rdg = BayesianRidge()
rdgreg = rdg.fit(xd_train, yd_train)


# In[109]:


# validation
print("Regresion Model Score" , ":" , rdgreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , rdgreg.score(xd_test, yd_test))

yd_predicted = rdgreg.predict(xd_train)
yd_test_predicted = rdgreg.predict(xd_test)

print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))


# ## 4.3. LARS Lasso Regression

# In[110]:


from sklearn.linear_model import LassoLars
las = LassoLars(alpha=0.3, fit_intercept=False, normalize=True)
lasreg = las.fit(xd_train, yd_train)


# In[111]:


print("Regresion Model Score" , ":" , lasreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , lasreg.score(xd_test, yd_test))

yd_predicted = lasreg.predict(xd_train)
yd_test_predicted = lasreg.predict(xd_test)

print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))


# ## 4.4. Decision Tree Regression

# In[112]:


from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(min_samples_leaf=20)
treereg = tree.fit(xl_train, yl_train)


# In[113]:


print("Regresion Model Score" , ":" , treereg.score(xl_train, yl_train) , "," ,
      "Out of Sample Test Score" ,":" , treereg.score(xl_test, yl_test))

yl_predicted = treereg.predict(xl_train)
yl_test_predicted = treereg.predict(xl_test)
print("Training RMSE", ":", rmse(yl_train, yl_predicted),
      "Testing RMSE", ":", rmse(yl_test, yl_test_predicted))
print("Training MAPE", ":", mape(yl_train, yl_predicted),
      "Testing MAPE", ":", mape(yl_test, yl_test_predicted))


# ## 4.5. Random Forest Regression

# In[114]:


from sklearn.ensemble import RandomForestRegressor
rdf = RandomForestRegressor(n_estimators=30)
rdfreg = rdf.fit(xl_train, yl_train)


# In[115]:


print("Regresion Model Score" , ":" , rdfreg.score(xl_train, yl_train) , "," ,
      "Out of Sample Test Score" ,":" , rdfreg.score(xl_test, yl_test))   

yl_predicted = rdfreg.predict(xl_train)
yl_test_predicted = rdfreg.predict(xl_test)

print("Training RMSE", ":", rmse(yl_train, yl_predicted),
      "Testing RMSE", ":", rmse(yl_test, yl_test_predicted))
print("Training MAPE", ":", mape(yl_train, yl_predicted),
      "Testing MAPE", ":", mape(yl_test, yl_test_predicted))


# ## 4.6. K-Nearest Neighbors Regression

# In[116]:


from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 30)
knnreg = knn.fit(xd_train, yd_train)


# In[117]:


print("Regresion Model Score" , ":" , knnreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , knnreg.score(xd_test, yd_test))

yd_predicted = knnreg.predict(xd_train)
yd_test_predicted = knnreg.predict(xd_test)

print("Training RMSE", ":", rmse(yd_train, yd_predicted),
      "Testing RMSE", ":", rmse(yd_test, yd_test_predicted))
print("Training MAPE", ":", mape(yd_train, yd_predicted),
      "Testing MAPE", ":", mape(yd_test, yd_test_predicted))


# In[118]:


for x in range(1,30):
    knn = KNeighborsRegressor(n_neighbors = x)
    knnreg = knn.fit(xd_train, yd_train)
    print("Regresion Model Score" , ":" , knnreg.score(xd_train, yd_train) , "," ,
      "Out of Sample Test Score" ,":" , knnreg.score(xd_test, yd_test))


# ## 5. Feature Importance

# In[119]:


features = xl_train.columns
importances = rdfreg.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(8,10))
plt.title('Feature Importances', fontsize=20)
plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# In[ ]:





# In[ ]:




