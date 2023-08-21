from flask import Flask,render_template,url_for,request
import pandas as pd
import joblib
import numpy as np
import pickle
import tensorflow as tf
import keras

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = request.form.get('thal')
        
        # Preprocess categorical features
        sex_encoded = sex
        cp_encoded = cp
        fbs_encoded = fbs
        restecg_encoded = restecg
        exang_encoded = exang
        ca_encoded = ca
        
        # Preprocess string categorical feature
        thal_encoded = thal

         # Preprocess numerical features
        age_encoded = age
        trestbps_encoded = trestbps
        chol_encoded = chol
        thalach_encoded = thalach
        oldpeak_encoded = oldpeak
        slope_encoded = slope
        
        data = {
            "sex": np.array([[sex_encoded]]),
            "cp": np.array([[cp_encoded]]),
            "fbs": np.array([[fbs_encoded]]),
            "restecg": np.array([[restecg_encoded]]),
            "exang": np.array([[exang_encoded]]),
            "ca": np.array([[ca_encoded]]),
            "thal": np.array([[thal_encoded]]),
            "age": np.array([[age_encoded]]),
            "trestbps": np.array([[trestbps_encoded]]),
            "chol": np.array([[chol_encoded]]),
            "thalach": np.array([[thalach_encoded]]),
            "oldpeak": np.array([[oldpeak_encoded]]),
            "slope": np.array([[slope_encoded]]),
        }
        
        my_prediction = model.predict(data)
    


        
        
        return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)