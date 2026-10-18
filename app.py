import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="SC2 Early-Game Predictor", layout="wide")

st.title("🎮 StarCraft II Early-Game Predictor")
st.caption("Dissertation project: early-game telemetry and match outcome prediction")

MODEL_DIR = "./models"
DATA_DIR = "./data"

# -----------------------------
# Utility functions
# -----------------------------

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
                st.warning(f"Failed to load {path}: {exc}")

    return bundle


def find_explainer(model):
    if model is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None


def build_legacy_input_df(apm, worker_count, buildings, army_units, unspent_resources):
    return pd.DataFrame([
        {
            "apm": apm,
            "worker_count": worker_count,
            "buildings_constructed": buildings,
            "army_units": army_units,
            "unspent_resources": unspent_resources,
        }
    ])


def build_dual_frame_input_df(apm, worker_count, buildings, army_units, unspent_resources):
    return pd.DataFrame([
        {
            "p1_apm": apm,
            "p1_workers": worker_count,
            "p1_buildings": buildings,
            "p1_army": army_units,
            "p1_unspent": unspent_resources,
        }
    ])


def explain_instance(model, explainer, df, selected_mode):
    if explainer is None:
        return None

    try:
        explanation = explainer(df)
        if hasattr(explanation, "values"):
            vals = explanation.values
            if len(np.asarray(vals).shape) == 3:
                vals = vals[:, :, 1]
            return explanation, vals
        arr = np.asarray(explanation)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        return explanation, arr
    except Exception:
        return None, None


# -----------------------------
# Load models
# -----------------------------
model_bundle = load_model_bundle()
model_names = [name for name, model in model_bundle.items() if model is not None]

if not model_names:
    st.error("No trained model artifacts were found in ./models. Please run the training pipeline first.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Model artifact", model_names)
model = model_bundle[selected_model_name]
explainer = find_explainer(model)

# -----------------------------
# Sidebar: user inputs
# -----------------------------
st.sidebar.header("Predicted match state")
st.sidebar.caption("Set the early-game telemetry values used for prediction.")

apm = st.sidebar.slider("APM (Actions per Minute)", 10, 800, 180)
worker_count = st.sidebar.slider("Worker Count", 0, 200, 45)
buildings = st.sidebar.slider("Buildings Constructed", 0, 60, 10)
army_units = st.sidebar.slider("Army Units", 0, 500, 15)
unspent_resources = st.sidebar.slider("Unspent Resources", 0, 10000, 450)

# -----------------------------
# Prediction modes
# -----------------------------
if selected_model_name == "single_frame":
    input_df = build_legacy_input_df(apm, worker_count, buildings, army_units, unspent_resources)
    mode = "Legacy single-frame model"
else:
    input_df = build_dual_frame_input_df(apm, worker_count, buildings, army_units, unspent_resources)
    mode = "Dual-frame model"

# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Prediction")
    try:
        proba = float(model.predict_proba(input_df)[0, 1])
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)
        prediction = "WIN" if proba >= threshold else "LOSS"

        st.metric(label="Predicted outcome", value=prediction, delta=f"{proba * 100:.1f}% Win probability")
        st.progress(float(proba))
        st.write(f"Model: {selected_model_name} | Mode: {mode}")
    except Exception as exc:
        st.error(f"Prediction failed. Please verify the model feature columns match the app inputs.\n{exc}")
        st.stop()

with right:
    st.subheader("Local SHAP explanation")
    if explainer is not None:
        explanation, vals = explain_instance(model, explainer, input_df, selected_model_name)
        if explanation is not None and vals is not None:
            try:
                fig, ax = plt.subplots(figsize=(7, 4))
                shap.plots.waterfall(explanation[0], show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                try:
                    feature_names = input_df.columns.tolist()
                    values = np.asarray(vals).reshape(-1)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.barh(feature_names, values)
                    ax.set_title("SHAP values (local)")
                    ax.invert_yaxis()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as inner_exc:
                    st.warning(f"SHAP visualisation could not be rendered: {inner_exc}")
        else:
            st.info("SHAP explanation is available but could not be extracted for the current model schema.")
    else:
        st.info("This model artifact has no compatible SHAP explainer.")

st.divider()

# -----------------------------
# Detailed table and model explainability
# -----------------------------
with st.container():
    st.subheader("Current input features")
    st.dataframe(input_df, use_container_width=True)

    if selected_model_name == "single_frame":
        st.write("This model expects the legacy single-frame schema: apm, worker_count, buildings_constructed, army_units, unspent_resources.")
    else:
        st.write("This model expects the dual-frame schema: p1_apm, p1_workers, p1_buildings, p1_army, p1_unspent.")

# -----------------------------
# Optional reports view
# -----------------------------
reports = [
    os.path.join("./reports", file)
    for file in os.listdir("./reports")
    if os.path.isfile(os.path.join("./reports", file))
] if os.path.exists("./reports") else []

if reports:
    st.subheader("Available outputs")
    for path in reports:
        st.write(path)

st.sidebar.markdown("---")
st.sidebar.write("Project focus: early-game predictability in professional StarCraft II matches.")
