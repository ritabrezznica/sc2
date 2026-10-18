import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="SC2 Early-Game Predictor", layout="wide")

st.title("🎮 StarCraft II Early-Game Strategy Predictor (t = 5 min)")
st.write("Dissertation Pipeline: Machine Learning Classification & SHAP Explainability")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("./models/sc2_rf_model.pkl")
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_artifacts()

# Sidebar User Inputs
st.sidebar.header("Input Match Telemetry (t = 5 min)")
apm = st.sidebar.slider("APM (Actions Per Minute)", 30, 400, 180)
worker_count = st.sidebar.slider("Worker Count", 12, 80, 45)
buildings = st.sidebar.slider("Buildings Constructed", 2, 25, 10)
army_units = st.sidebar.slider("Army Units", 0, 50, 15)
unspent_resources = st.sidebar.slider("Unspent Resources (Minerals + Gas)", 0, 2000, 450)

# Build Input DataFrame
input_df = pd.DataFrame([{
    'apm': apm,
    'worker_count': worker_count,
    'buildings_constructed': buildings,
    'army_units': army_units,
    'unspent_resources': unspent_resources
}])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Model Prediction")
    proba = model.predict_proba(input_df)[0][1]
    prediction = "WIN" if proba >= 0.5 else "LOSS"
    
    st.metric(label="Predicted Outcome", value=prediction, delta=f"{proba*100:.1f}% Win Probability")
    st.progress(float(proba))

with col2:
    st.subheader("Local SHAP Attribution")
    shap_vals = explainer(input_df)
    
    if len(shap_vals.shape) == 3:
        vals = shap_vals[:, :, 1]
    else:
        vals = shap_vals
        
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(vals[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.subheader("Telemetry Matrix Overview")
st.dataframe(input_df, use_container_width=True)