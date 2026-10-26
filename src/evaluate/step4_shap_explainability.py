import os
import joblib
import numpy as np
import pandas as pd
import shap

print("--> Loading models for Dual-Frame SHAP valuation...")
model_a = joblib.load("./models/sc2_rf_frame_a.pkl")
model_b = joblib.load("./models/sc2_rf_frame_b.pkl")

df = pd.read_csv("./data/processed/dual_frame_match_features.csv")

X_a = df[['p1_apm', 'p1_workers', 'p1_buildings', 'p1_army', 'p1_unspent']]
X_b = df[['delta_apm', 'delta_workers', 'delta_buildings', 'delta_army', 'delta_unspent']]

explainer_a = shap.TreeExplainer(model_a)
explainer_b = shap.TreeExplainer(model_b)

shap_a = explainer_a(X_a)
shap_b = explainer_b(X_b)

vals_a = shap_a[:, :, 1] if len(shap_a.shape) == 3 else shap_a
vals_b = shap_b[:, :, 1] if len(shap_b.shape) == 3 else shap_b

df_a = pd.DataFrame({
    'Frame A Feature': X_a.columns,
    'Mean |SHAP| (Frame A)': np.abs(vals_a.values).mean(axis=0)
}).sort_values(by='Mean |SHAP| (Frame A)', ascending=False)

df_b = pd.DataFrame({
    'Frame B Feature': X_b.columns,
    'Mean |SHAP| (Frame B)': np.abs(vals_b.values).mean(axis=0)
}).sort_values(by='Mean |SHAP| (Frame B)', ascending=False)

print("\n================ FRAME A SHAP RANKING ================")
print(df_a.to_string(index=False))

print("\n================ FRAME B SHAP RANKING ================")
print(df_b.to_string(index=False))
print("======================================================\n")