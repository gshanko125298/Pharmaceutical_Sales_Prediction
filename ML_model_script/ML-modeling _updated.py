#!/usr/bin/env python
# coding: utf-8

# # Modleing and prediction using Machine lerining and Deep learining

# ##### An end-to-end Data Science project with a regression adapted for time series as solution was created four machine learning models to forecast the sales. 

# In[1]:


import math
import pandas                as pd
import numpy                 as np
import seaborn               as sns
import matplotlib.pyplot     as plt
import inflection
import datetime
import warnings
import random
import pickle
import json

import xgboost               as xgb
    
from tabulate                import tabulate
from pandas.api.types        import is_string_dtype, is_numeric_dtype
from matplotlib              import gridspec
from scipy                   import stats as ss
from sklearn.preprocessing   import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error
from sklearn.linear_model    import LinearRegression, Lasso


from IPython.core.display    import HTML
from IPython.display         import Image
# Versão da Linguagem Python
from platform                import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())
warnings.filterwarnings( 'ignore' )


# ### Function used as supporter 

# In[2]:


def jupyter_settings():
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('pylab', 'inline')
    
    plt.style.use( 'bmh' )
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    
    display( HTML( '<style>.container { width:100% !important; }</style>') )
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.set_option( 'display.expand_frame_repr', False )
    
    sns.set()
    
    def cramer_v( x, y ):
        cm = pd.crosstab( x, y ).values # Confusion Matrix
        n = cm.sum()
        r, k = cm.shape

        chi2 = ss.chi2_contingency( cm )[0]
        chi2corr = max( 0, chi2 - (k-1)*(r-1)/(n-1) )

        kcorr = k - (k-1)**2/(n-1)
        rcorr = r - (r-1)**2/(n-1)

        return np.sqrt( (chi2corr/n) / ( min( kcorr-1, rcorr-1 ) ) )

def mean_absolute_percentage_error( y, yhat ):
    return np.mean( np.abs( ( y-yhat ) / y ))

def mean_percentage_error( y, yhat ):
    return np.mean( ( y - yhat ) / y )

def ml_error ( model_name, y, yhat):
    mae = mean_absolute_error( y,yhat )
    mape = mean_absolute_percentage_error( y,yhat )
    rmse = np.sqrt(mean_squared_error( y,yhat ))
    
    return pd.DataFrame( {'Model Name': model_name,
                          'MAE': mae,
                          'MAPE': mape,
                          'RMSE': rmse}, index=[0])
# time-series cross validation implementation
def cross_validation( x_training, kfold, model_name, model, verbose=False ):
    mae_list = []
    mape_list = []
    rmse_list = []
     
    for k in reversed( range( 1, kfold+1 ) ): #k-fold implementation
        if verbose:
            print( '\nKFold Number: {}'.format( k ) )
        # start and end date for validation 
        start_date_validation = x_training['date'].max() - datetime.timedelta( days=k*6*7)  
        end_date_validation = x_training['date'].max() - datetime.timedelta( days=(k-1)*6*7) 

        # filtering dataset
        training = x_training[x_training['date'] < start_date_validation]
        validation = x_training[(x_training['date'] >= start_date_validation) & (x_training['date'] <= end_date_validation)]

        # training and validation dataset
        # training
        xtraining = training.drop( ['date', 'sales'], axis=1 ) 
        ytraining = training['sales']

        # validation
        xvalidation = validation.drop( ['date', 'sales'], axis=1 )
        yvalidation = validation['sales']

        # model
        m = model.fit( xtraining, ytraining )

        # prediction
        yhat = m
        # performance
        m_result = ml_error( model_name, np.expm1( yvalidation ), np.expm1( yhat ) )

        # store performance of each kfold iteration
        mae_list.append(  m_result['MAE'] )
        mape_list.append( m_result['MAPE'] )
        rmse_list.append( m_result['RMSE'] )

    return pd.DataFrame( {'Model Name': model_name,
                          'MAE CV': np.round( np.mean( mae_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mae_list ), 2 ).astype( str ),
                          'MAPE CV': np.round( np.mean( mape_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( mape_list ), 2 ).astype( str ),
                          'RMSE CV': np.round( np.mean( rmse_list ), 2 ).astype( str ) + ' +/- ' + np.round( np.std( rmse_list ), 2 ).astype( str ) } )


# In[3]:


jupyter_settings()


# ### loading of dataset from local repo.

# In[4]:


df_train_data = pd.read_csv(r"C:\Users\Genet Shanko\Pharmaceutical_Sales_Prediction\DVC_Dataset\train.csv")
df_store_data = pd.read_csv(r"C:\Users\Genet Shanko\Pharmaceutical_Sales_Prediction\DVC_Dataset\Store.csv")

# Merge


# In[5]:


df_train_data.head()


# In[6]:


df_store_data.head()


# In[7]:


df_train_store = pd.merge( df_train_data, df_store_data,how='left', on = 'Store' )


# In[8]:


df_train_store.head()


# ## DATA DESCRIPTION

# In[9]:


#Copy dataset
df1 =df_train_store.copy()


# ### Rename Columns

# In[10]:


cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']

snakecase = lambda x: inflection.underscore(x)

cols_new = list( map( snakecase, cols_old ) )

#Rename Columns

df1.columns = cols_new


# In[11]:


df1.columns


# In[12]:


##### Data Dimension and Data type


# In[13]:


print( 'Number of Rows: {}'.format( df1.shape[0] ) )
print( 'Number of Cols: {}'.format( df1.shape[1] ) )


# In[14]:


#  transform datatype of the variable data to datetime
df1['date'] = pd.to_datetime( df1[ 'date' ] )
df1.info()


# ### missing values

# In[15]:


# checking NA values sum
missing_count = df1.isnull().sum() # the count of missing values
value_count = df1.isnull().count() # the count of all values

missing_percentage = round(missing_count/value_count *100, 2) # the percentage of missing values
missing_df = pd.DataFrame({'missing values count': missing_count, 'percentage': missing_percentage})
missing_df


# In[16]:


barchart = missing_df.plot.bar(y='percentage')
for index, percentage in enumerate( missing_percentage ):
    barchart.text( index, percentage, str(percentage)+'%')


# ### Filling the missing values

# In[17]:


# competition_distance
df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan( x ) else x )

# competition_open_since_month
df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] ) else x['competition_open_since_month'],
                                               axis = 1 )
# competition_open_since_year
df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] ) else x['competition_open_since_year'],
                                               axis = 1 )
# promo2_since_week
df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] ) else x['promo2_since_week'],
                                               axis = 1 )
# promo2_since_year
df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] ) else x['promo2_since_year'],
                                               axis = 1 )
# promo_interval
month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5:'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

df1['promo_interval'].fillna( 0, inplace = True)
df1['month_map'] = df1['date'].dt.month.map( month_map )
#df1['is_promo'] = df1[[ 'promo_interval', 'month_map' ]].apply(lambda x: 0 if x['promo_interval'] == 0 else x['month_map'] in x['promo_interval'])


# In[18]:


df1.isna().sum()


# In[19]:


df1.dtypes


# ### Change Types

# In[20]:


f1 = df1.astype({'competition_open_since_month':'int64', 'competition_open_since_year':'int64', 'promo2_since_week':'int64', 'promo2_since_year':'int64' })


# In[21]:


df1.dtypes # checking datatypes transformation


# ### Descriptive Statistics

# In[22]:


df1.describe().T


# In[23]:


# separate numerical and categorical attributes

num_attributes = df1.select_dtypes( include = 'number')
cate_attributes = df1.select_dtypes( include = 'object')


# ### Numerical Attributes

# In[24]:


# Central Tendency - Mean, median
ct1 = pd.DataFrame( num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame( num_attributes.apply( np.median ) ).T

# Dispersion - std, min, max, range, skew, kurtoisis
d1 = pd.DataFrame(num_attributes.apply( np.std )).T
d2 = pd.DataFrame(num_attributes.apply( min )).T
d3 = pd.DataFrame(num_attributes.apply( max )).T
d4 = pd.DataFrame(num_attributes.apply( lambda x: x.max() - x.min() )).T
d5 = pd.DataFrame(num_attributes.apply( lambda x: x.skew() )).T
d6 = pd.DataFrame(num_attributes.apply( lambda x: x.kurtosis() )).T

#concatenate
m = pd.concat( [d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()


m.columns = ['attributes', 'min', 'max', 'range', 'mean', 'median', 'std', 'skew', 'kurtosis']
m


# ### Categorical Attributes

# In[25]:


# check unique values of categorical features
cate_attributes.apply( lambda x: x.unique().shape[0])


# In[26]:


# plot boxplots of categorical features against target variable
aux1 = df1[(df1['state_holiday'] != '0') & (df1['sales'] > 0)]

plt.subplot (1, 3, 1)
sns.boxplot(x='state_holiday', y= 'sales', data=aux1);

plt.subplot (1, 3, 2)
sns.boxplot(x='store_type', y= 'sales', data=aux1);

plt.subplot (1, 3, 3)
sns.boxplot(x='assortment', y= 'sales', data=aux1);


# ### Feature Engineering

# In[29]:


df2 = df1.copy()


# In[34]:


df2.head().T


# In[35]:


df3 = df2.copy()


# In[36]:


df3 = df3[(df3['open'] != 0) & (df3['sales']> 0)]


# In[37]:


cols_drop = ['customers', 'open', 'promo_interval', 'month_map']
df3 = df3.drop( cols_drop, axis=1 )


# In[38]:


df3.head()


# ### EXPLORATORY DATA ANALYSIS (EDA)

# In[39]:


df4 = df3.copy()


# In[40]:


fig = plt.figure( figsize = (14, 6), constrained_layout=True)
sns.distplot(df4['sales'], kde = False);


# #### Numerical Variable

# In[41]:


num_attributes.hist(bins = 25);


# ### Categorical Variable

# In[42]:


# state_holiday
plt.subplot(3, 2, 1)
a = df4[df4['state_holiday'] != 'regular_day']
sns.countplot(data = a, x = a['state_holiday'])

plt.subplot(3, 2, 2)
sns.kdeplot(data = df4, x = df4[df4['state_holiday'] == 'public_holiday']['sales'], shade=True )
sns.kdeplot(data = df4, x = df4[df4['state_holiday'] == 'easter_holiday']['sales'], label='easter_holiday', shade=True)
sns.kdeplot(data = df4, x = df4[df4['state_holiday'] == 'Christmas']['sales'], label='Christmas', shade=True )

# store_type
plt.subplot(3, 2, 3)
sns.countplot(data = df4, x = df4['store_type'])

plt.subplot(3, 2, 4)
sns.kdeplot(data = df4, x = df4[df4['store_type'] == 'a']['sales'], shade = True)
sns.kdeplot(data = df4, x = df4[df4['store_type'] == 'b']['sales'], shade = True)
sns.kdeplot(data = df4, x = df4[df4['store_type'] == 'c']['sales'], shade = True)
sns.kdeplot(data = df4, x = df4[df4['store_type'] == 'd']['sales'], shade = True)

# assortment
plt.subplot(3, 2, 5)
sns.countplot(data = df4, x = df4['assortment'])

plt.subplot(3, 2, 6)
sns.kdeplot(data = df4, x = df4[df4['assortment'] == 'basic']['sales'], shade = True)
sns.kdeplot(data = df4, x = df4[df4['assortment'] == 'extended']['sales'], shade = True)
sns.kdeplot(data = df4, x = df4[df4['assortment'] == 'extra']['sales'], shade = True);


# ### Bivariate Analysis

# In[43]:


aux1 = df4[['assortment', 'sales']].groupby('assortment').sum().reset_index()
sns.barplot(x = 'assortment', y = 'sales', data = aux1)

aux2 = df4[['year_week','assortment', 'sales']].groupby(['year_week','assortment']).sum().reset_index()
aux2.pivot( index = 'year_week', columns = 'assortment', values = 'sales').plot()


# In[44]:


aux1 = df4[['competition_distance', 'sales']].groupby('competition_distance').sum().reset_index()

plt.subplot(3, 1, 1)
sns.scatterplot(x= 'competition_distance', y= 'sales', data= aux1);

plt.subplot(3, 1, 2)
bins = list(np.arange( 0, 20000, 1000))
aux1['competition_distance_binned'] = pd.cut(aux1['competition_distance'], bins = bins)
aux2 = aux1[['competition_distance_binned', 'sales']].groupby('competition_distance_binned').sum().reset_index()
plt.xticks(rotation = 20)
sns.barplot( x= 'competition_distance_binned', y= 'sales', data = aux2);

plt.subplot(3, 1, 3)
sns.heatmap(aux1.corr(method= 'pearson'), annot= True);


# In[45]:


df4[['promo', 'promo2', 'sales']].groupby(['promo', 'promo2']).sum().reset_index().sort_values(by='sales', ascending = True)


# In[46]:


aux1 = df4[( df4['promo'] ==1 ) & (df4['promo2'] == 1)][['year_week', 'sales']].groupby('year_week').sum().reset_index()
ax = aux1.plot()

aux2 = df4[( df4['promo'] ==1 ) & (df4['promo2'] == 0)][['year_week', 'sales']].groupby('year_week').sum().reset_index()
aux2.plot( ax = ax )

ax.legend( labels = ['Tradicional & Extendida', 'Extendida']);


# ### sales in Christmas hoday is more

# In[47]:


aux = df4[df4['state_holiday'] != 'regular_day']

plt.subplot(1, 2, 1)
aux1 = aux[['state_holiday', 'sales']].groupby('state_holiday').sum().reset_index()
sns.barplot(data = aux1, x= 'state_holiday', y= 'sales');

plt.subplot(1, 2, 2)
aux2 = aux[['year', 'state_holiday','sales']].groupby(['year', 'state_holiday']).sum().reset_index()
sns.barplot(data = aux2, x = 'year', y= 'sales', hue= 'state_holiday');


# ### Stores should sell more over the years.

# In[48]:


aux = df4[['year', 'sales']].groupby('year').sum().reset_index()

plt.subplot(131)
sns.barplot(data = aux, x= 'year', y= 'sales');

plt.subplot(132)
sns.regplot(data = aux, x= 'year', y= 'sales');
plt.xticks(rotation = 90)

plt.subplot(133)
sns.heatmap(aux.corr(method= 'pearson'), annot = True);


# ### Stores should sell more in the second half of the year.

# In[49]:


aux = df4[['month', 'sales']].groupby('month').sum().reset_index()

plt.subplot(131)
sns.barplot(data = aux, x= 'month', y= 'sales');   ###*tentar colocar legenda no gráfico em cima do semestre que vende menos*###

plt.subplot(132)
sns.regplot(data = aux, x= 'month', y= 'sales');
plt.xticks(rotation = 90)

plt.subplot(133)
sns.heatmap(aux.corr(method= 'pearson'), annot = True);


# ### Stores should sell less on weekends.

# In[50]:


aux = df4[['day_of_week', 'sales']].groupby('day_of_week').sum().reset_index()

plt.subplot(131)
sns.barplot(data = aux, x= 'day_of_week', y= 'sales');

plt.subplot(132)
sns.regplot(data = aux, x= 'day_of_week', y= 'sales');
plt.xticks(rotation = 90)

plt.subplot(133)
sns.heatmap(aux.corr(method= 'pearson'), annot = True);


# ### Stores should sell less during school holidays

# In[51]:


aux = df4[['school_holiday', 'sales']].groupby('school_holiday').sum().reset_index();

fig = plt.figure(figsize = (18,12))
plt.subplot(211)
sns.barplot(data = aux, x= 'school_holiday', y= 'sales');

aux2 = df4[['month','school_holiday', 'sales']].groupby(['month','school_holiday']).sum().reset_index();

plt.subplot(212)
sns.barplot(data = aux2, x= 'month', y= 'sales', hue= 'school_holiday');
plt.xticks(rotation = 45);


# ### Summary 

# In[52]:


summary = pd.DataFrame({'Hypothesis':['Stores with extended assortment type sell more.',
                                      'Stores with near competitors sell less.',
                                      'Stores with longer competitors should sell more.',
                                      'Stores with longer active promo should sell more.',
                                      'Stores with more promotion days should sell more.',
                                      'Stores with more consecutive promotions should sell more.',
                                      'Stores open during the Christmas holiday should sell more.',
                                      'Stores should sell more over the years.',
                                      'Stores should sell more in the second half of the year.',
                                      'Stores should sell more after the 10th of each month.',
                                      'Stores should sell less on weekends.',
                                      'Stores should sell less during school holidays.',
                                     ],
                        'True / False':['False', 'False', 'False', 'False', '-', 'False', 'False', 'False', 'False',
                                        'True','True', 'True'], 
                        'Relevance':['Low', 'Medium', 'Medium', 'Low', '-', 'Low', 'Medium', 'High', 'High', 
                                     'High', 'High', 'Low']}, 
                        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
summary


# #### Multivariate Analysis

# In[53]:


correlation = (num_attributes.corr( method = 'pearson' ))
sns.heatmap( correlation, annot = True );


# ## Categorical Attributes

# ### DATA PREPARATION

# In[54]:


df5 = df4.copy()


# In[55]:


# boxplot to check outliers sensitivity
sns.boxplot(x = df5['competition_distance']);


# In[56]:


df5.select_dtypes( include = 'object').head()


# In[57]:


df5['sales'] = np.log1p( df5['sales'] )
sns.distplot(df5['sales']);


# In[64]:


# day of week
df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

# month
df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi/12 ) ) )
df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi/12 ) ) )

# day 
df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

# week of year
df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )


# In[67]:


#df5 = df5.astype({'store_type': 'int64'})
df5.dtypes


# In[68]:


df5.head()


# In[69]:


df6 = df5.copy()


# In[70]:


# Spliting dataframe into training and test. 
# Test will have the last 6 weeks of sales (i want to predict the next 6 weeks, and because of using time series i can make a randomic selection)
# starting at 2015-06-19 until the last day of sales

df6[['store', 'date']].groupby('store').max().reset_index()['date'][0] - datetime.timedelta( days = 6*7 )


# In[71]:


# training dataset
X_train = df6[df6['date'] < '2015-06-19']
y_train = X_train['sales']

# test dataset
X_test = df6[df6['date'] >= '2015-06-19']
y_test = X_test['sales']

print( 'Training Min Date: {}'.format( X_train['date'].min() ) )
print( 'Training Max Date: {}'.format( X_train['date'].max() ) )

print( '\nTest Min Date: {}'.format( X_test['date'].min() ) )
print( 'Test Max Date: {}'.format( X_test['date'].max() ) )


# ### Feature selection 

# In[72]:


cols_selected_boruta = ['store',
 'promo',
 'store_type',
 'assortment',
 'competition_distance',
 'competition_open_since_month',
 'competition_open_since_year',
 'promo2',
 'promo2_since_week',
 'promo2_since_year',
 'competition_time_month',
 'promo_time_week',
 'day_of_week_sin',
 'day_of_week_cos',
 'month_cos',
 'month_sin',
 'day_sin',
 'day_cos',
 'week_of_year_cos',
 'week_of_year_sin']

# columns to add
feat_to_add = ['date', 'sales']
# final features

cols_selected_boruta_full = cols_selected_boruta.copy()
cols_selected_boruta_full.extend( feat_to_add )


# In[73]:


pd.DataFrame(data = cols_selected_boruta, columns = ['feature_selected'])


# ### MACHINE LEARNING ALGORITHM MODELS

# In[74]:


type(cols_selected_boruta_full)


# In[87]:


import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# In[97]:


df6['error'] = df6['sales'] - df6['predictions']
df6['error_rate'] = df6['predictions'] / df6['sales']
plt.subplot( 2, 2, 1 )
sns.lineplot( x='date', y='sales', data=df6, label='SALES' )
sns.lineplot( x='date', y='predictions', data=df6, label='PREDICTIONS' )

plt.subplot( 2, 2, 2 )
sns.lineplot( x='date', y='error_rate', data=df6 )
plt.axhline( 1, linestyle='--', color = 'red')

plt.subplot( 2, 2, 3 )
sns.distplot( df6['error'] )

plt.subplot( 2, 2, 4 )
sns.scatterplot( df6['predictions'], df6['error'] )


# In[ ]:





# In[ ]:




