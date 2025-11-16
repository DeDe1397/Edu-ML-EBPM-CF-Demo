import os, joblib, pandas as pd, numpy as np, streamlit as st
from google.cloud import storage
from modules.config import GCS_BUCKET, MODEL_PATHS, FEATURE_PATH
import shap

def _create_bg(feature_list, n=100):
    data={}
    for f in feature_list:
        if f in ["reading_score","writing_score"]:
            data[f]=np.random.uniform(50,100,n)
        else:
            data[f]=np.random.choice([0,1],n,p=[0.8,0.2])
    return pd.DataFrame(data, columns=feature_list)

@st.cache_resource
def load_model_artifacts(model_name:str):
    """GCSからモデル/特徴量を取得 → ローカルキャッシュ → SHAP Explainer生成"""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    local_model = f".cache_{model_name}.pkl"
    if not os.path.exists(local_model):
        bucket.blob(MODEL_PATHS[model_name]).download_to_filename(local_model)
    model = joblib.load(local_model)

    if not os.path.exists(".cache_feature_list.pkl"):
        bucket.blob(FEATURE_PATH).download_to_filename(".cache_feature_list.pkl")
    feature_list = joblib.load(".cache_feature_list.pkl")

    bg = _create_bg(feature_list, 120)
    explainer = shap.TreeExplainer(model, bg) if model_name=="LightGBM" else shap.LinearExplainer(model, bg)
    return model, feature_list, explainer
