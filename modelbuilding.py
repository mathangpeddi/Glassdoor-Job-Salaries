# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
import os
os.chdir('/home/user/Courses/Machine Learning/Projects/KenJee')

df=pd.read_csv('eda_data.csv')

# Choose relevant columns
df.columns
df.shape
df_model=df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp',
             'hourly','employer_provided','job_state','same_state','age','python_yn','spark','aws','excel',
             'job_simp','seniority','desc_len']]

# If we have categorical data then we need to create dummy variables for the columns(this obviously increases the no of columns in our dataframe)
# For eg.if we have male and female then for each row it checks if the value is male or female-if its male then it represents it with 1 and female with 0 as so on 
df_dum=pd.get_dummies(df_model)  # after this the no of columns have increased from 20 to 178
df.shape

# train test split
from sklearn.model_selection import train_test_split
X=df_dum.drop('avg_salary',axis=1)
y=df_dum['avg_salary'].values 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) # 80% in our training set and 20% in our test set

# multiple linear regression-here we do multiple linear regession in statsmodels as well as scikit-learn just to compare them

# Using statsmodels

import statsmodels.api as sm

X_sm= X = sm.add_constant(X)
model=sm.OLS(y,X_sm) # OLS stands for Ordinary Least Squares 
model.fit().summary() # gives a summary of all the variables 
# R-squared value is 70% so it shows 70% variation in the glassdoor salaries,num_comp coeff is 2.25 which means for each additioanl competitor we are adding 2250 dollars to the salary
# We even observe that there is some sort of collinearity between industry and sector 
# When it comes to the jobs everything except data engineer is relevant(if the p-value is more than 0.05 then its not relevant)

# Using scikit-learn

from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import cross_val_score

lm=LinearRegression()
lm.fit(X_train,y_train)

np.mean(cross_val_score(lm,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) # so the mean is 20 which means 20K dollars

# Lasso regression-this dataset is going to be sparse with all these dummy variables which helps us normalize that
# So as there is limited data so lasso regression goes through the data and normalizes it which is better for our model
lm_l=Lasso(alpha=0.13)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) # now when it starts then it becomes a little bit worse(21K dollars)

alpha=[]
error=[]

for i in range(1,100):
    alpha.append(i/100) # here we can take i/10 as well but the error is still high so that's why we reduce the values of alpha
    lm_l=Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)))
plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['alpha','error'])
df_err[df_err.error == max(df_err.error)] # so the alpha value is 0.13 is giving the best error term
# We can also improve the model tuning the GridSearch 

# Random forest(we can even use support vector regression,XGBoost and any other models) 
# Random Forest Regression-Its a tree based decision process and also there are many 0s,1s so we expect it to be a better model
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) # So here we are getting a smaller value of error than the previous one

# Tune models using GridSearchCV-you put in all the parameters which you want,it runs all the models and splits the ones with best results(that's why we make of GridSearch)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')} 
# estimators-10 to 300,criteterion-Mean Square Error and Mean Absolute Error

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_ # so we have improved our model(before the error was 16 and now it reduced down to 14 which is pretty good)
gs.best_estimator_ # tells us about all the parameters of our model 

# test ensembles-so now we have to predict the test set data
tpred_lm=lm.predict(X_test) # using linear regression 
tpred_lml=lm_l.predict(X_test) # using lasso regression
tpred_rf=gs.best_estimator_.predict(X_test) # using random forest regression

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm) 
mean_absolute_error(y_test,tpred_lml)
mean_absolute_error(y_test,tpred_rf) # so the mean absolute error for this is less when compared to both the models

# Saving model to disk
pickle.dump(gs,open('model.pkl','wb')) # So this file model.pkl has to be deployed in Heroku environent which is a Platform As A Service  
# Pickle library will help us dump the model.
# Dump the model in the form of an extension .pkl and it will be dumped in write bytes mode(wb)

# Loading the model to compare the results
model=pickle.load(open('model.pkl','rb')) # Read it in read bytes(rb) mode

 
# Sometimes its better to combine different models and predict so that it can increase our performance 
mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2) # so the error may or may not increase becuase one model might be overtraining
# So the tuned random forest model is the best compared to all becuase it has the least error 
# So instead of taking the average we can even take like 90% of random forest model and 10% of any other model and test our accuracy/performance
# mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2)-these type of ensemble models are better for classification problems
# Generally lasso regression should have more effect than linear regression as it has the normalization effect and we have kind of a sparse matrix but in this case the lasso performed worse than the linear regression.(so it depends on model to model-we cannot generalize anything)

