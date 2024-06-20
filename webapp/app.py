from flask import Flask, request
from flask import render_template
import xgboost as xgb

import pandas as pd
app = Flask(__name__)

import joblib
cuts = ["Good", "Ideal", "Premium", "Very Good"]
clarities = ["IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"]
colors =["E","F", "G", "H", "I", "J"]

@app.route("/")
def index():
    return render_index()

def render_index(**params):
        
    cut_select = f"""
        <select id="cut" name="cut" required>
            {''.join([f'<option value="{i}">{i}</option>' for i in cuts])}
        </select>
     """
    color_select =  f"""
        <select id="color" name="color" required>
            {''.join([f'<option value="{i}">{i}</option>' for i in colors])}
        </select>
    """
    
    clarity_select =  f"""
        <select id="clarity" name="clarity" required>
            {''.join([f'<option value="{i}">{i}</option>' for i in clarities])}
        </select>
    """
    params.update({"cut_select": cut_select, "color_select": color_select, "clarity_select": clarity_select})
    return render_template('index.html',**params)

@app.route("/submit", methods=['GET', 'POST'])
def predict_diamond_price():
    data = {
        'carat' : [float(request.form.get('carat'))],
        'cut': [request.form.get('cut')],
        'color': [request.form.get('color')],
        'clarity' : [request.form.get('clarity')],
        'depth' : [float(request.form.get('depth'))],
        'table' : [float(request.form.get('table'))],
        'x': [float(request.form.get('x'))],
        'y': [float(request.form.get('y'))],
        'z': [float(request.form.get('z'))]
    }

    

    model = load_best_model()
    df = pd.DataFrame(xgboost_preprocess(data)) 
    price = round(float(model.predict(df)[0]), 2)

    
    return render_index(predicted_price = price)


def xgboost_preprocess(df):
    df["cut"] = pd.Categorical(df['cut'], categories=cuts, ordered=True)
    df['color'] = pd.Categorical(df['color'], categories=colors, ordered=True)
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarities, ordered=True)
    return df

def load_best_model():
    return joblib.load("best_model/XGBoost-trials100.pkl")
