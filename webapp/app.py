from flask import Flask, request
from flask import render_template


import pandas as pd
from utils import *

app = Flask(__name__)
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

@app.route("/submit", methods=['POST'])
def submit():
    action = request.form.get('action')
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
    n_diamond = request.form.get('ndiamond')
    if not n_diamond:
        n_diamond = 10
    else:
        n_diamond = int(n_diamond)
    if action == 'predict':
        return predict_diamond_price(data)
    elif action == 'similar':
        return get_similar_diamonds(data, n_diamond)
    else:
        return "<p>Unknown action!</p>"


def predict_diamond_price(data):
    model = load_best_model()
    df = pd.DataFrame(xgboost_preprocess(data, cuts=cuts, clarities=clarities, colors=colors)) 
    price = round(float(model.predict(df)[0]), 2)
    return render_index(predicted_price = price)

def get_similar_diamonds(data, n_sample):
    print(n_sample)
    df = read_df("data/diamonds.csv")
    filtered_df = df[
        (df['cut'] == data['cut'][0]) &
        (df['color'] == data['color'][0]) &
        (df['clarity'] == data['clarity'][0])
    ]
    
    
   
    filtered_df['carat_diff'] = (filtered_df['carat'] - data['carat'][0]).abs()
    most_similar_rows = filtered_df.nsmallest(n_sample, 'carat_diff')

    table_html = most_similar_rows.to_html(classes='table table-striped', index=True)
    
    return render_index(diamond_data = table_html)
