#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:44:46 2021

@author: adititelang
"""


import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

userhome = os.path.expanduser('~')
csvfile = userhome + r'/Desktop/Python/End2EndProjects/Data/Netflix.csv'
df = pd.read_csv(csvfile)

X = df[['High', 'Low', 'Close']].values
y = df['Open'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('R2 Score : {}'.format(r2_score(y_test, y_pred)))


joblib.dump(model, userhome + r'/Desktop/Python/End2EndProjects/Netflix_API/netflix_model.pkl')