import pandas as pd
import joblib

def xgboost_preprocess(df, cuts, colors, clarities):
    df["cut"] = pd.Categorical(df['cut'], categories=cuts, ordered=True)
    df['color'] = pd.Categorical(df['color'], categories=colors, ordered=True)
    df['clarity'] = pd.Categorical(df['clarity'], categories=clarities, ordered=True)
    return df

def load_best_model():
    return joblib.load("best_model/XGBoost-trials100.pkl")
