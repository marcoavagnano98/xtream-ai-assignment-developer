from flask import Flask, request, render_template
import pandas as pd
from utils import *

app = Flask(__name__)

CUTS = ["Good", "Ideal", "Premium", "Very Good"]
CLARITIES = ["IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"]
COLORS = ["E", "F", "G", "H", "I", "J"]

@app.route("/")
def index():
    return render_index()

def render_index(**params):
    params.update({
        "cut_select": create_select("cut", CUTS),
        "color_select": create_select("color", COLORS),
        "clarity_select": create_select("clarity", CLARITIES)
    })
    return render_template('index.html', **params)

def create_select(name, options):
    return f"""
        <select id="{name}" name="{name}" required>
            {''.join([f'<option value="{i}">{i}</option>' for i in options])}
        </select>
    """

@app.route("/submit", methods=['POST'])
def submit():
    action = request.form.get('action')
    data = request.form.to_dict(flat=True)
    try:
        data = {
            'carat': [float(data['carat'])],
            'cut': [data['cut']],
            'color': [data['color']],
            'clarity': [data['clarity']],
            'depth': [float(data['depth'])],
            'table': [float(data['table'])],
            'x': [float(data['x'])],
            'y': [float(data['y'])],
            'z': [float(data['z'])]
        }
        
        n_diamond = int(data.get('ndiamond', 10))

        if action == 'predict':
            return predict_diamond_price(data)
        if action == 'similar':
            return get_similar_diamonds(data, n_diamond)
    except:
        return '<center><h2>Insert numbers in form !</h2></center>'



def predict_diamond_price(data):
    model = load_best_model()
    df = pd.DataFrame(xgboost_preprocess(data, cuts=CUTS, clarities=CLARITIES, colors=COLORS))
    price = round(float(model.predict(df)[0]), 2)
    return render_index(predicted_price=price)

def get_similar_diamonds(data, n_sample):
    df = read_df("data/diamonds.csv")
    filtered_df = df[
        (df['cut'] == data['cut'][0]) &
        (df['color'] == data['color'][0]) &
        (df['clarity'] == data['clarity'][0])
    ]
    
    filtered_df['carat_diff'] = (filtered_df['carat'] - data['carat'][0]).abs()
    most_similar_rows = filtered_df.nsmallest(n_sample, 'carat_diff')
    table_html = most_similar_rows.to_html(classes='table table-striped', index=True)
    
    return render_index(diamond_data=table_html)

if __name__ == "__main__":
    app.run(debug=True)
