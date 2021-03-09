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
import os 

app = Flask(__name__, template_folder='templates')

@app.route("/")

@app.route("/amazon")
def stock():
    return render_template("amazon.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list)
    userhome = os.path.expanduser('~')
    loaded_model = joblib.load(userhome + r'/Desktop/Python/End2EndProjects/Amazon_API/amazon_model.pkl')
    result = loaded_model.predict(to_predict)
    return result[0]
  

@app.route('/predict', methods = ["POST"])
def predict():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
         #diabetes
        if(len(to_predict_list)==7):
            result = ValuePredictor(to_predict_list,7)
    
    if(int(result)==1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return(render_template("result.html", prediction_text=prediction))       

if __name__ == "__main__":
    app.run(debug=True)