#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:57:53 2021

@author: adititelang
"""

import pandas as pd
import os 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib

userhome = os.path.expanduser('~')
csvfile = userhome + r'/Desktop/Python/End2EndProjects/Data/Netflix.csv'
df = pd.read_csv(csvfile)


df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')


Min_date = df.index.min()
Max_date = df.index.max()
print ("First date is",Min_date)
print ("Last date is",Max_date)
print (Max_date - Min_date)


df = df.drop(columns=['High', 'Low', 'Close', 'Adj Close', 'Volume'])

num = 20 # forcasting 20 days ahead
df['label'] = df['Open'].shift(-num) # forcasting open column

Data = df.drop(['label'],axis=1)

X = Data.values
X = preprocessing.scale(X) 
X = X[:-num]  # X will contain everything except last num items after scaling

df.dropna(inplace=True)
Target = df.label
y = Target.values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test , y_test) #Returns the coefficient of determination R^2 of the prediction

X_Predict = X[-num:] # last num items in array
Forecast = model.predict(X_Predict)


joblib.dump((model,X_Predict, df), userhome + r'/Desktop/Python/End2EndProjects/Netflix_API/netflix_forecast_model.pkl')