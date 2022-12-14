# Pharmaceutical Sales Prediction across multiple stores using Deep learning 

 the project was conducted for 10 Academy Batch 6 - Weekly Challenge of  Week-3

**Table of content**

- Introduction
- learining outcome
- Install
- Data
- EDA
- Models using Machine learning
- Models using Deep learning
- Conclusion


## Introduction

 The project aims to forcust sales of six week, ahead of time for Rossmann Pharmaceuticals as a Machine Learning Engineer.
 this project used to predict by using fourstore datasets.

##Learning Outcomes

> Statistical Modelling

> Using core data science python libraries pandas, matplotlib, seaborn, scikit-learn

> ML algorithms Logistic regression, Decision Trees, XGBoost

> Model management (building ML catalog contains model feature labels and training model version)

> MLOps with DVC, CML, and MLFlow

## Install

```
git clone https://github.com/Ad-Campaign-Performance/SmartAd_A-B_Testing_user_analysis.git
cd Pharmaceutical_Sales_Prediction
pip install -r requirements.txt
```

## Data and features

- The BIO data for this project :Train , test and Store Data 

Data can be found [here at google drive](https://[[drive.google.com/drive/u/0/folders/1rbiLJuE6WQzqX1sxhXsd1VtwwDumKeGE]]

The data collected for this challenge has the following columns

-Id - an Id that represents a (Store, Date) duple within the test set
- auction_id: the unique id of the online user who has been presented the BIO.
- Store - a unique Id for each store
- Sales - the turnover for any given day (this is what you are predicting
- date: the date in YYYY-MM-DD format
- Customers - the number of customers on a given day
- Open - an indicator for whether the store was open: 0 = closed, 1 = open
- StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays.
- StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. 
- SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
- StoreType - differentiates between 4 different store models: a, b, c, d
- Assortment - describes an assortment level: a = basic, b = extra, c = extended. Read more about assortment here
- CompetitionDistance - distance in meters to the nearest competitor store
- CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
- Promo - indicates whether a store is running a promo on that day
- Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
- Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
- PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. 

## EDA

>  This steps are necessary both for feature selection and machine learning models.  All the Data Processing, statistical and sequential Analysiwas sone in this steps  that can be used as a data anlysis, data  understanding  , data visualization by observing the relation between some the features with target features. 
> some the EDA as shown below 
> Univariate Analysis
> Sales 
> ![image](https://user-images.githubusercontent.com/43541659/189528154-5ae06b44-8d97-402f-8db3-1eafbe270eed.png)
>Bivariate Analysis
>Sales Over Time
>![image](https://user-images.githubusercontent.com/43541659/189529866-2f6cf9b3-2aca-40ed-ad23-3419a53736a3.png)
>Sales By Promotion
>![image](https://user-images.githubusercontent.com/43541659/189530196-5ff31cad-8c3d-4035-89ad-5d7578e7dedc.png)

## Models using machine learning

>  Machine learning models were trained. The one that presented best results on training and testing results are selected  after cross-validation was conducted.
> 
## Model using Deep learning

> Using deep learning models the analysis conducted and best result was taken.

## Conclusion

> Based on the above model analysis the Random Forest Regression analysis shows better results. Based on the EDA analysis the following points are observed :
- Stores open during the Christmas holiday should sell more.
- Stores should sell less on weekends.
- Stores should sell less during school holidays


## Authors
???? **Genet Shanko**

- GitHub: [Genet Shanko](https://github.com/gshanko125298))
- LinkedIn: [Genet Shanko](https://www.linkedin.com/in/genet-dekebo-24b34658/)
- Website: [Genet Shanko Demo Porfolio](https://github.com/Ad-Campaign-Performance/SmartAd_A-B_Testing_user_analysis/pull/11)


