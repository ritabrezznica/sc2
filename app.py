import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="SC2 Early-Game Predictor", layout="wide")

st.title("🎮 StarCraft II Early-Game Predictor")
st.caption("Dissertation project: early-game telemetry and match outcome prediction")

MODEL_DIR = "./models"

@st.cache_resource
def load_model_bundle():
    bundle = {
        "single_frame": None,
        "dual_frame_a": None,
        "dual_frame_b": None,
    }
    model_paths = {
        "single_frame": os.path.join(MODEL_DIR, "sc2_rf_model.pkl"),
        "dual_frame_a": os.path.join(MODEL_DIR, "sc2_rf_frame_a.pkl"),
        "dual_frame_b": os.path.join(MODEL_DIR, "sc2_rf_frame_b.pkl"),
    }

    for key, path in model_paths.items():
        if os.path.exists(path):
            try:
                bundle[key] = joblib.load(path)
            except Exception as exc:
                st.sidebar.warning(f"Failed to load {path}: {exc}")
    return bundle

def build_legacy_input_df(apm, worker_count, buildings, army_units, unspent_resources):
    return pd.DataFrame([{
        "apm": apm,
        "worker_count": worker_count,
        "buildings_constructed": buildings,
        "army_units": army_units,
        "unspent_resources": unspent_resources
    }])

def build_frame_a_input_df(apm, worker_count, buildings, army_units, unspent_resources):
    return pd.DataFrame([{
        "p1_apm": apm,
        "p1_workers": worker_count,
        "p1_buildings": buildings,
        "p1_army": army_units,
        "p1_unspent": unspent_resources
    }])

def build_frame_b_input_df(p1, p2):
    return pd.DataFrame([{
        "delta_apm": p1["apm"] - p2["apm"],
        "delta_workers": p1["workers"] - p2["workers"],
        "delta_buildings": p1["buildings"] - p2["buildings"],
        "delta_army": p1["army"] - p2["army"],
        "delta_unspent": p1["unspent"] - p2["unspent"]
    }])

def safe_shap_waterfall(model, input_df):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_df)
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            shap.plots.waterfall(shap_values[0], show=False)
        except Exception:
            vals = shap_values.values if hasattr(shap_values, "values") else np.array(shap_values)
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            names = input_df.columns
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(names, vals[0])
            ax.invert_yaxis()
            ax.set_title("SHAP values (local)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"SHAP not available: {e}")

model_bundle = load_model_bundle()
available_models = [k for k, v in model_bundle.items() if v is not None]

if not available_models:
    st.error("No trained model artifacts were found in ./models. Please run the training pipeline first.")
    st.stop()

selected_model = st.sidebar.selectbox("Model artifact", available_models)

st.sidebar.header("Predicted match state")
st.sidebar.caption("Set early-game telemetry values used for prediction.")

def slider_int(label, minv, maxv, default):
    return st.sidebar.slider(label, minv, maxv, default)

if selected_model == "dual_frame_b":
    st.sidebar.subheader("Player 1 telemetry")
    p1_apm = slider_int("P1 APM", 10, 800, 180)
    p1_workers = slider_int("P1 Worker Count", 0, 200, 45)
    p1_buildings = slider_int("P1 Buildings Constructed", 0, 60, 10)
    p1_army = slider_int("P1 Army Units", 0, 500, 15)
    p1_unspent = slider_int("P1 Unspent Resources", 0, 10000, 450)

    st.sidebar.subheader("Player 2 telemetry")
    p2_apm = slider_int("P2 APM", 10, 800, 170)
    p2_workers = slider_int("P2 Worker Count", 0, 200, 40)
    p2_buildings = slider_int("P2 Buildings Constructed", 0, 60, 8)
    p2_army = slider_int("P2 Army Units", 0, 500, 12)
    p2_unspent = slider_int("P2 Unspent Resources", 0, 10000, 300)

    p1 = {"apm": p1_apm, "workers": p1_workers, "buildings": p1_buildings, "army": p1_army, "unspent": p1_unspent}
    p2 = {"apm": p2_apm, "workers": p2_workers, "buildings": p2_buildings, "army": p2_army, "unspent": p2_unspent}
    input_df = build_frame_b_input_df(p1, p2)
    schema_note = "frame B (delta)"

elif selected_model == "dual_frame_a":
    p1_apm = slider_int("APM (Actions per Minute)", 10, 800, 180)
    p1_workers = slider_int("Worker Count", 0, 200, 45)
    p1_buildings = slider_int("Buildings Constructed", 0, 60, 10)
    p1_army = slider_int("Army Units", 0, 500, 15)
    p1_unspent = slider_int("Unspent Resources", 0, 10000, 450)
    input_df = build_frame_a_input_df(p1_apm, p1_workers, p1_buildings, p1_army, p1_unspent)
    schema_note = "frame A (absolute)"

else:
    apm = slider_int("APM (Actions per Minute)", 10, 800, 180)
    worker_count = slider_int("Worker Count", 0, 200, 45)
    buildings = slider_int("Buildings Constructed", 0, 60, 10)
    army_units = slider_int("Army Units", 0, 500, 15)
    unspent_resources = slider_int("Unspent Resources", 0, 10000, 450)
    input_df = build_legacy_input_df(apm, worker_count, buildings, army_units, unspent_resources)
    schema_note = "legacy single-frame"

left, right = st.columns(2)

with left:
    st.subheader("Prediction")
    try:
        model = model_bundle[selected_model]
        proba = float(model.predict_proba(input_df)[0, 1])
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)
        prediction = "WIN" if proba >= threshold else "LOSS"
        st.metric(label="Predicted outcome", value=prediction, delta=f"{proba * 100:.1f}% Win probability")
        st.progress(float(proba))
        st.write(f"Model: {selected_model} | Schema: {schema_note}")
    except Exception as e:
        st.error(f"Prediction failed. Check model / input schema.\n{e}")

with right:
    st.subheader("Local SHAP explanation")
    try:
        safe_shap_waterfall(model, input_df)
    except Exception as e:
        st.info(f"SHAP unavailable: {e}")

st.divider()
st.subheader("Input features")
st.dataframe(input_df, use_container_width=True)

if os.path.exists("./reports"):
    files = sorted([f for f in os.listdir("./reports") if os.path.isfile(os.path.join("./reports", f))])
    if files:
        st.subheader("Reports")
        for f in files:
            st.write(f"./reports/{f}")

st.sidebar.markdown("---")
st.sidebar.write("Project: early-game predictability (SC2).")