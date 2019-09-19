# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:59:14 2017

@author: Hou LAN 41423018



###############################################################################
#######################  Decision Tree & Linear Regression  ###################
###############################################################################


"""


import pandas as pd
import numpy as np






#*********************************************#
#
#    Decision Tree, Random Forest & Gradient Boosting
#
#*********************************************#



#####################################
#>>> 1> Read file

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

print(titanic.head())#Pre-look at titanic as a dataframe
print(titanic.info())#Look at titanic statistic information




#####################################
#>>> 2> Select features and Vectorize

X = titanic[['pclass','age','sex']]
y = titanic['survived']


print(X.info()) # Check features information

X['age'].fillna(X['age'].mean(),inplace=True)

print(X.info()) # Check again after filling out
  

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33) #Split data

vec = DictVectorizer(sparse=False)#Vectorize

X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)
X_test=vec.transform(X_test.to_dict(orient='record'))


#####################################
#>>> 3> Build model and Predict

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict=dtc.predict(X_test)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)



#####################################
#>>> 4> Score
from sklearn.metrics import classification_report

# Decision Tree
print('The accuracy of Decision Tree is',dtc.score(X_test,y_test))
print(classification_report(y_predict,y_test))


# Random Forest
print('The accuracy of Random Forest Classifier is',rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test))


# Gradient Boosting
print('The accuracy of Gradient Boosting Classifier is',gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))





#*********************************************#
#
#    Logistic Regression & SGD Regression
#
#*********************************************#


#####################################
#>>> 1> Read file

from sklearn.datasets import load_boston
boston=load_boston()
print(boston.DESCR)




#####################################
#>>> 2> Split data and preprocessing

# Split data
from sklearn.cross_validation import train_test_split

X=boston.data
y=boston.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33) #Split data
print('The max target value is', np.max(boston.target))
print('The min target value is', np.min(boston.target))
print('The average target value is', np.mean(boston.target))

# preprocessing
from sklearn.preprocessing import StandardScaler

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.fit_transform(X_test)

y_train=ss_y.fit_transform(y_train)
y_test=ss_y.fit_transform(y_test)



#####################################
#>>> 3> Build model and Predict

# Linear Regression
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

# SGDRegressor
from sklearn.linear_model import SGDRegressor
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_pred=sgdr.predict(X_test)



#####################################
#>>> 4> Score


# Linear Regression
print('The value of default measurement of LinearRegression is',lr.score(X_test,y_test))

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

# R-squared
print('The value of R-squared of LinearRegression is',r2_score(y_test,lr_y_predict))


# Mean squared error
print('The value of mean squared error of LinearRegression is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))


# Mean absolute error
print('The value of mean absolute error of LinearRegression is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))







#*********************************************#
#
#          SVM
#
#*********************************************#


#####################################
#>>> 1> Read file

titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')


#####################################
#>>> 2> Vectorize

X = titanic.drop(['row.names','name','survived'],axis=1)
y = titanic['survived']


#fill missing value
X['age'].fillna(X['age'].mean(),inplace=True)
X.fillna('UNKNOWN',inplace=True)
  

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=33) #Split data

vec = DictVectorizer(sparse=False)

X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print(len(vec.feature_names_))
X_test=vec.transform(X_test.to_dict(orient='record'))


#####################################
#>>> 3> Build model

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
print(dt.score(X_test,y_test))



#####################################
#>>> 4> Select feature and predict
from sklearn import feature_selection

fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)

X_train_fs=fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)

X_test_fs=fs.transform(X_test)
print(dt.score(X_test_fs,y_test))

#####################################
#>>> 5> Validation

from sklearn.cross_validation import cross_val_score
percentiles=range(1,100,2)
results=[]

for i in percentiles:
    fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=i)
    X_train_fs=fs.fit_transform(X_train,y_train)
    scores=cross_val_score(dt,X_train_fs,y_train,cv=5)
    results=np.append(results,scores.mean())  

print(results)

opt=np.where(results==results.max())[0]
print('Optimal number of feature %d' %percentiles[opt])# sth wrong with this line

    

#####################################
#>>> 6> Visualization

import pylab as pl
pl.plot(percentiles,results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

#Validation
from sklearn import feature_selection


fs=feature_selection.SelectPercentile(feature_selection.chi2,percentile=20)

X_train_fs=fs.fit_transform(X_train,y_train)
dt.fit(X_train_fs,y_train)

X_test_fs=fs.transform(X_test)
print(dt.score(X_test_fs,y_test))



#*********************************************#
#
#          Regularization
#
#*********************************************#


#linearregression
#####################################
#>>> 1> Input dataset
X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]


#####################################
#>>> 2> Build linearregression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#####################################
#>>> 3> Predict
xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)
yy=regressor.predict(xx)

#####################################
#>>> 4> Visualization and Output
import matplotlib.pyplot as plt


plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1])
plt.show()


print('The R-squared value of Linear Regressor performing on the training data is',regressor.score(X_train,y_train))



#polynominal featrues in degree 2
#####################################
#>>> 2&3>  Build model and Predict
from sklearn.preprocessing import PolynomialFeatures

poly2=PolynomialFeatures(degree=2)
X_train_poly2=poly2.fit_transform(X_train)

regressor_poly2=LinearRegression()
regressor_poly2.fit(X_train_poly2,y_train)

xx_poly2=poly2.transform(xx)
yy_poly2=regressor_poly2.predict(xx_poly2)


#####################################
#>>> 4> Visualization and Output
plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1,plt2])
plt.show()  

print('The R-squared value of Polynominal Regressor(Degree=2) performing on the training data is',regressor_poly2.score(X_train_poly2,y_train))


#polynominal featrues in degree 4
#####################################
#>>> 2&3>  Build model and Predict
from sklearn.preprocessing import PolynomialFeatures

poly4=PolynomialFeatures(degree=4)
X_train_poly4=poly4.fit_transform(X_train)

regressor_poly4=LinearRegression()
regressor_poly4.fit(X_train_poly4,y_train)

xx_poly4=poly4.transform(xx)
yy_poly4=regressor_poly4.predict(xx_poly4)
print(regressor_poly4.coef_)
print(np.sum(regressor_poly4.coef_**2))

#####################################
#>>> 4> Visualization and Output
plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label='Degree=2')
plt4,=plt.plot(xx,yy_poly4,label='Degree=4')
plt.axis([0,25,0,25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1,plt2,plt4])
plt.show()  

print('The R-squared value of Polynominal Regressor(Degree=4) performing on the training data is',regressor_poly4.score(X_train_poly4,y_train))


#####################################
#>>> 5> Validation

#Prepare test dataset
X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]

#linear regression
print('The R-squared value of Linear Regressor performing on the test data is',regressor.score(X_test,y_test))

#polynominal 2
X_test_poly2=poly2.fit_transform(X_test)
print('The R-squared value of Polynominal Regressor(Degree=2) performing on the test data is',regressor_poly2.score(X_test_poly2,y_test))

#polynominal 4
X_test_poly4=poly4.fit_transform(X_test)
print('The R-squared value of Polynominal Regressor(Degree=4) performing on the test data is',regressor_poly4.score(X_test_poly4,y_test))



#lasso
#####################################
#>>> 2&3>  Build model and Predict
from sklearn.linear_model import Lasso

lasso_poly4=Lasso()
lasso_poly4.fit(X_train_poly4,y_train)
print(lasso_poly4.score(X_test_poly4,y_test))
print(lasso_poly4.coef_)


#ridge
#####################################
#>>> 2&3>  Build model and Predict
from sklearn.linear_model import Ridge

ridge_poly4=Ridge()
ridge_poly4.fit(X_train_poly4,y_train)
print(ridge_poly4.score(X_test_poly4,y_test))
print(ridge_poly4.coef_)
print(np.sum(ridge_poly4.coef_**2))


