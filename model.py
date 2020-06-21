import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle

df=pd.read_csv('eda_data.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()

df['job_simp'].unique()
df.dtypes

df['Rating']=df['Rating'].astype(float)
df_model=df[['avg_salary','Rating','age','job_simp']]

df_dum=pd.get_dummies(df_model) 
df_dum.head()

from sklearn.model_selection import train_test_split
X=df_dum.drop('avg_salary',axis=1)
y=df_dum['avg_salary'].values 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#  Lasso Regression

# =============================================================================
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import cross_val_score
# 
# X_train.columns

#  =============================================================================
#  lml=Lasso()
#  lml.fit(X_train,y_train)
#  np.mean(cross_val_score(lml,X_train,y_train,scoring='neg_mean_absolute_error',cv=3))
#  
#  from sklearn.metrics import accuracy_score-This is wrong as we shud use lm.score for regression problems
#  lml_pred=lml.predict(X_test)  # accuracy_score is used only for classification problems
#  accuracy_score(y_test, lml_pred)
#  
#  pickle.dump(lml,open('model1.pkl','wb'))
#  =============================================================================
# =============================================================================

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


lm=LinearRegression()
lm.fit(X_train,y_train)

np.mean(cross_val_score(lm,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)) 


lm_pred=lm.predict(X_test)
lm.score(X_test,y_test)

pickle.dump(lm, open('model2.pkl','wb'))
df_model.dtypes
model1 = pickle.load(open('model1.pkl','rb')) # model1 is using Lasso Regression
model2 = pickle.load(open('model2.pkl','rb')) # model2 is using Linear Regression







