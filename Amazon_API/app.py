#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:30:57 2021

@author: adititelang
"""

from flask import Flask, render_template, url_for, flash, redirect
import joblib
from flask import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import sys

app = Flask(__name__, template_folder='templates')
userhome = os.path.expanduser('~')

@app.route("/")

@app.route("/amazon")
def stock():
    return render_template("amazon.html")

def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    loaded_model = joblib.load(userhome + r'/Desktop/Python/End2EndProjects/Amazon_API/amazon_model.pkl')
    result = loaded_model.predict(to_predict)
    return result[0]
  

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
    
        result = ValuePredictor(to_predict_list, 3)
        return(render_template("result.html", prediction_text= "%0.4f" % result)) # to print upto 4 decimal place     
   
@app.route('/plot', methods = ["POST"])    
def plot():
    if request.method == "POST":
        loaded_model, X_Predict, df = joblib.load(userhome + r'/Desktop/Python/End2EndProjects/Amazon_API/amazon_forecast_model.pkl')
        num = request.form['noOfDays']
        start_date_of_forecasting = request.form['Date']
        #hard coded value to 20 days as model trained for 20 days and any other number gives shape error
        trange = pd.date_range(start_date_of_forecasting, periods=int(num), freq='d')  
        loaded_forecast_model = loaded_model.predict(X_Predict)
        Predict_df = pd.DataFrame(loaded_forecast_model, index=trange)
        Predict_df.columns = ['Forecast']
        df_concat = pd.concat([df, Predict_df], axis=1)
        df_concat['Forecast'].plot(color='orange', linewidth=3)
        plt.xlabel('Time')
        plt.ylabel('Price')
        img_addr = userhome + r'/Desktop/Python/End2EndProjects/Amazon_API/static/img/plot.png'
        plt.savefig(img_addr)
        return render_template('plot.html')
        
    
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response
    
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True