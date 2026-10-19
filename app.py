from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# StarCraft II Match Intelligence — Streamlit application
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Match Intelligence | StarCraft II",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DEFAULT = "final_sc2_winner_model.joblib"
DATA_DEFAULT = "sc2_paired_training.csv"
REPORTED_METRICS = {
    "Accuracy": 0.5302,
    "Balanced accuracy": 0.5199,
    "ROC-AUC": 0.5391,
}

# White canvas with vivid blue, purple, pink, teal and orange accents.
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172033; --muted:#667085; --line:#E7EAF2; --blue:#2864F0; --purple:#7856FF; --pink:#EC4899; --teal:#0EA5A4; --orange:#F59E0B; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#FFFFFF; color:var(--ink); }
[data-testid="stHeader"] { background:rgba(255,255,255,.92); }
.block-container { max-width:1450px; padding-top:2.1rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#FBFCFF; border-right:1px solid #E8EBF3; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#172033 !important; letter-spacing:-.035em; }
h1 { font-weight:800 !important; }
h2,h3 { font-weight:750 !important; }
p,li { color:#4B5568; }
.eyebrow { color:#2864F0; font-size:.72rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.45rem; }
.hero { background:linear-gradient(115deg,#F4F7FF 0%,#FFFFFF 55%,#FFF5FB 100%); border:1px solid #E6EAF7; border-radius:24px; padding:30px 32px; position:relative; overflow:hidden; margin:0 0 24px 0; }
.hero:after { content:''; position:absolute; width:210px; height:210px; border-radius:50%; right:-70px; top:-95px; background:linear-gradient(135deg,rgba(40,100,240,.13),rgba(236,72,153,.10)); }
.hero-kicker { color:#2864F0; font-size:.75rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; margin-bottom:10px; }
.hero-title { color:#172033; font-family:'Manrope',sans-serif; font-size:clamp(1.7rem,3vw,2.6rem); line-height:1.12; font-weight:800; letter-spacing:-.05em; margin-bottom:12px; }
.hero-copy { max-width:760px; font-size:1rem; line-height:1.7; color:#596579; margin-bottom:0; }
.metric-card { border:1px solid #E8EBF3; border-radius:17px; padding:19px 20px; background:#fff; min-height:118px; box-shadow:0 4px 18px rgba(25,39,75,.025); }
.metric-label { color:#737D90; font-size:.78rem; font-weight:700; margin-bottom:12px; }
.metric-value { color:#172033; font-family:'Manrope',sans-serif; font-size:1.75rem; line-height:1.1; font-weight:800; letter-spacing:-.04em; }
.metric-note { color:#8A93A4; font-size:.76rem; margin-top:8px; }
.panel { border:1px solid #E8EBF3; border-radius:18px; padding:20px 22px; background:#fff; }
.panel-title { font-family:'Manrope',sans-serif; color:#172033; font-size:1.05rem; font-weight:800; margin-bottom:5px; }
.panel-copy { color:#7A8496; font-size:.84rem; line-height:1.55; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#10B981; margin-right:7px; }
.small-note { color:#7A8496; font-size:.8rem; line-height:1.6; }
div[data-testid="stFormSubmitButton"] button, .stButton button { border-radius:11px; font-weight:700; min-height:2.7rem; }
.stButton button[kind="primary"], div[data-testid="stFormSubmitButton"] button { background:#2864F0; border:1px solid #2864F0; color:white; }
.stButton button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] button:hover { background:#174ED4; border-color:#174ED4; }
[data-testid="stMetric"] { background:#fff; border:1px solid #E8EBF3; padding:15px 17px; border-radius:15px; }
[data-testid="stMetricLabel"] { color:#6F7A8D; }
[data-testid="stMetricValue"] { color:#172033; font-family:'Manrope',sans-serif; font-weight:800; }
hr { border-color:#E8EBF3; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def resolve_path(value: str) -> Path:
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    return Path.cwd() / p


def load_resources(model_path: str, data_path: str):
    model, data, model_error, data_error = None, None, None, None
    mp, dp = resolve_path(model_path), resolve_path(data_path)
    try:
        if mp.exists():
            model = joblib.load(mp)
        else:
            model_error = f"Model file not found: {mp}"
    except Exception as exc:
        model_error = f"Could not load model: {exc}"
    try:
        if dp.exists():
            data = pd.read_csv(dp, low_memory=False)
        else:
            data_error = f"Dataset file not found: {dp}"
    except Exception as exc:
        data_error = f"Could not load dataset: {exc}"
    return model, data, model_error, data_error


def get_expected_features(model, data):
    """Find the original input columns expected by the fitted estimator."""
    if model is not None:
        for obj in (model, getattr(model, "named_steps", {}).get("preprocessor"),
                    getattr(model, "named_steps", {}).get("prep")):
            names = getattr(obj, "feature_names_in_", None) if obj is not None else None
            if names is not None and len(names):
                return list(names)
        # A ColumnTransformer often exposes the original columns through its transformers.
        try:
            pre = model.named_steps.get("preprocessor") or model.named_steps.get("prep")
            if pre is not None and hasattr(pre, "transformers_"):
                cols = []
                for _, _, selected in pre.transformers_:
                    if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                        cols.extend(list(selected))
                if cols:
                    return list(dict.fromkeys(cols))
        except Exception:
            pass
    if data is not None:
        excluded = {"player1_win", "replay_index", "replay_id", "match_id", "map_name", "map"}
        possible = [c for c in data.columns if c not in excluded]
        if "player1_race" in possible and "player2_race" in possible:
            numeric = [c for c in possible if c not in ("player1_race", "player2_race") and pd.api.types.is_numeric_dtype(data[c])]
            return numeric + ["player1_race", "player2_race"]
        return possible
    return []


def clean_for_model(frame: pd.DataFrame, expected_features, data=None):
    out = frame.copy()
    for col in expected_features:
        if col not in out.columns:
            if col in ("player1_race", "player2_race"):
                out[col] = "Unknown"
            elif data is not None and col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                med = pd.to_numeric(data[col], errors="coerce").median()
                out[col] = 0.0 if pd.isna(med) else float(med)
            else:
                out[col] = 0.0
    out = out[expected_features].copy()
    for col in expected_features:
        if col in ("player1_race", "player2_race"):
            out[col] = out[col].fillna("Unknown").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fallback = 0.0
            if data is not None and col in data.columns:
                median = pd.to_numeric(data[col], errors="coerce").median()
                if not pd.isna(median):
                    fallback = float(median)
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(fallback)
    return out


def numeric_feature_columns(expected_features):
    return [c for c in expected_features if c not in ("player1_race", "player2_race")]


def pretty_feature(name: str) -> str:
    label = str(name)
    for prefix in ("player1_", "player2_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label.replace("_", " ").strip().title()


def sample_dropdown_values(series: pd.Series, max_values=35):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return [0.0]
    unique = np.sort(vals.unique())
    if len(unique) <= max_values:
        choices = unique.tolist()
    else:
        qs = np.linspace(0, 1, max_values)
        choices = np.unique(np.quantile(vals.to_numpy(), qs)).tolist()
    choices = [float(x) for x in choices if np.isfinite(x)]
    if not choices:
        choices = [float(vals.median())]
    return choices


def default_for_feature(feature, data):
    if data is None or feature not in data.columns:
        return 0.0
    val = pd.to_numeric(data[feature], errors="coerce").median()
    return 0.0 if pd.isna(val) else float(val)


def render_hero(kicker, title, copy):
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{kicker}</div>'
        f'<div class="hero-title">{title}</div><p class="hero-copy">{copy}</p></div>',
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# Sidebar configuration
with st.sidebar:
    st.markdown('<div class="eyebrow">SC2 ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("## Match Intelligence")
    st.caption("Early-game match analytics")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Match Prediction", "Player Comparison", "Model Evaluation", "About"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**Data connections**")
    model_path = st.text_input("Model file", value=MODEL_DEFAULT)
    data_path = st.text_input("Dataset file", value=DATA_DEFAULT)
    reload_data = st.button("Reload model and dataset", use_container_width=True)
    st.markdown('<p class="small-note">Predictions are estimates based on the available features, not guarantees.</p>', unsafe_allow_html=True)

cache_key = (str(resolve_path(model_path)), str(resolve_path(data_path)))
if "resource_key" not in st.session_state or st.session_state.resource_key != cache_key or reload_data:
    model, data, model_error, data_error = load_resources(model_path, data_path)
    st.session_state.resource_key = cache_key
    st.session_state.model = model
    st.session_state.data = data
    st.session_state.model_error = model_error
    st.session_state.data_error = data_error
else:
    model = st.session_state.model
    data = st.session_state.data
    model_error = st.session_state.model_error
    data_error = st.session_state.data_error

feature_names = get_expected_features(model, data)
num_features = len(feature_names)

# Identify race/category columns and match rows for optional data exploration.
race_features = [c for c in ("player1_race", "player2_race") if c in feature_names]
numeric_features = numeric_feature_columns(feature_names)

if page == "Overview":
    st.markdown('<div class="eyebrow">MATCH INTELLIGENCE / OVERVIEW</div>', unsafe_allow_html=True)
    st.title("StarCraft II match analytics")
    st.caption("Explore early-game statistics, match predictions, and model performance.")
    render_hero(
        "EARLY-GAME MODEL · ~60 SECONDS",
        "Understand the match before it unfolds.",
        "Explore paired player statistics, estimate the probability of a Player 1 victory, and review the model's measured performance. Predictions are estimates, not guarantees.",
    )
    n_matches = len(data) if data is not None else 0
    map_col = next((c for c in ("map_name", "map", "map_name_player1") if data is not None and c in data.columns), None)
    unique_maps = data[map_col].nunique() if map_col else None
    cols = st.columns(4)
    with cols[0]: render_metric_card("Paired matches", f"{n_matches:,}" if data is not None else "—", "Available in loaded dataset")
    with cols[1]: render_metric_card("Model features", f"{num_features}" if feature_names else "—", "Inputs expected by model")
    with cols[2]: render_metric_card("Unique maps", f"{unique_maps:,}" if unique_maps is not None else "Not available", "When map metadata is present")
    with cols[3]: render_metric_card("Snapshot", "~60 sec", "Early-game information")

    st.markdown("### Data connections")
    st.caption("Model and dataset status")
    c1, c2 = st.columns(2)
    with c1:
        if model is not None:
            st.success(f"Model loaded: {Path(resolve_path(model_path)).name}")
        else:
            st.error(model_error or "Model could not be loaded.")
    with c2:
        if data is not None:
            st.success(f"Dataset loaded: {Path(resolve_path(data_path)).name}")
        else:
            st.warning(data_error or "Dataset could not be loaded.")
    st.markdown("### Model performance snapshot")
    st.caption("Reported held-out test metrics from the current project evaluation.")
    mcols = st.columns(3)
    for col, (label, val) in zip(mcols, REPORTED_METRICS.items()):
        with col:
            render_metric_card(label, f"{val:.2%}", "Held-out test result")
    st.markdown("<p class='small-note'>These metrics are reported project results; they are not recalculated from the currently loaded dataset.</p>", unsafe_allow_html=True)

elif page == "Match Prediction":
    st.markdown('<div class="eyebrow">MATCH INTELLIGENCE / PREDICTION</div>', unsafe_allow_html=True)
    st.title("Match Prediction")
    st.caption("Estimate the probability of a Player 1 victory from early-game player statistics.")
    render_hero("PREDICTION WORKSPACE", "Configure a match snapshot.", "Choose race categories and select values for the player statistics. Dropdown choices are sampled from observed dataset values when available.")

    if model is None:
        st.error(model_error or "Load the trained model before making predictions.")
    elif not feature_names:
        st.error("Could not identify the input features expected by the saved model. Check that the correct trained pipeline is loaded.")
    else:
        input_values = {}
        with st.form("prediction_form"):
            left, right = st.columns(2)
            for col, player_title, player_prefix, race_name in [
                (left, "Player 1", "player1_", "player1_race"),
                (right, "Player 2", "player2_", "player2_race"),
            ]:
                with col:
                    st.markdown(f"### {player_title}")
                    player_features = [f for f in numeric_features if f.startswith(player_prefix)]
                    if race_name in feature_names:
                        race_options = ["Unknown", "Terran", "Zerg", "Protoss", "Random"]
                        existing = []
                        if data is not None and race_name in data.columns:
                            existing = [str(x) for x in data[race_name].dropna().unique()]
                        race_options = list(dict.fromkeys(existing + race_options))
                        input_values[race_name] = st.selectbox(f"{player_title} race", race_options, index=0, key=f"{race_name}_select")
                    with st.expander(f"{player_title} statistics ({len(player_features)})", expanded=True):
                        for feat in player_features:
                            vals = sample_dropdown_values(data[feat]) if data is not None and feat in data.columns else [0.0]
                            default = default_for_feature(feat, data)
                            idx = min(range(len(vals)), key=lambda i: abs(vals[i] - default)) if vals else 0
                            label = pretty_feature(feat)
                            input_values[feat] = st.selectbox(label, options=vals, index=idx, format_func=lambda x: f"{x:,.3f}".rstrip("0").rstrip("."), key=f"pred_{feat}")
            submitted = st.form_submit_button("Estimate match outcome", type="primary", use_container_width=True)

        if submitted:
            row = pd.DataFrame([{f: input_values.get(f, "Unknown" if f in race_features else 0.0) for f in feature_names}])
            try:
                model_input = clean_for_model(row, feature_names, data)
                prediction = int(np.asarray(model.predict(model_input)).ravel()[0])
                probability = None
                if hasattr(model, "predict_proba"):
                    probs = np.asarray(model.predict_proba(model_input))[0]
                    classes = list(getattr(model, "classes_", [0, 1]))
                    try:
                        probability = float(probs[classes.index(1)])
                    except (ValueError, IndexError):
                        probability = float(probs[-1])
                if probability is None:
                    st.info(f"Model prediction: {'Player 1 wins' if prediction == 1 else 'Player 1 loses'}.")
                else:
                    winner_label = "Player 1" if probability >= 0.5 else "Player 2"
                    st.markdown("---")
                    st.subheader("Prediction result")
                    result_cols = st.columns([1, 2])
                    with result_cols[0]:
                        render_metric_card("Estimated Player 1 win probability", f"{probability:.1%}", "Model probability estimate")
                    with result_cols[1]:
                        st.markdown(f"### Estimated advantage: {winner_label}")
                        st.progress(float(np.clip(probability, 0, 1)))
                        st.write(f"Player 2 estimated win probability: **{1-probability:.1%}**")
                        st.caption("This is a model estimate, not a guarantee of the match result.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that the saved model and selected feature schema match the current project files.")

elif page == "Player Comparison":
    st.markdown('<div class="eyebrow">MATCH INTELLIGENCE / COMPARISON</div>', unsafe_allow_html=True)
    st.title("Player Comparison")
    st.caption("Compare two players' early-game statistics using the paired-match dataset.")
    if data is None:
        st.warning(data_error or "Load the dataset to compare player statistics.")
    else:
        paired = [(c, "player2_" + c[len("player1_"):]) for c in data.columns if c.startswith("player1_") and "player2_" + c[len("player1_"):] in data.columns and pd.api.types.is_numeric_dtype(data[c]) and pd.api.types.is_numeric_dtype(data["player2_" + c[len("player1_"):]])]
        if not paired:
            st.info("No matching numeric player1_/player2_ feature pairs were found in the dataset.")
        else:
            labels = {pretty_feature(a): (a, b) for a, b in paired}
            selected = st.multiselect("Statistics to compare", options=list(labels.keys()), default=list(labels.keys())[:6])
            if selected:
                records = []
                for label in selected:
                    a, b = labels[label]
                    records.extend([
                        {"Statistic": label, "Player": "Player 1", "Mean value": pd.to_numeric(data[a], errors="coerce").mean()},
                        {"Statistic": label, "Player": "Player 2", "Mean value": pd.to_numeric(data[b], errors="coerce").mean()},
                    ])
                comp = pd.DataFrame(records)
                fig = px.bar(comp, x="Statistic", y="Mean value", color="Player", barmode="group", color_discrete_map={"Player 1":"#2864F0", "Player 2":"#EC4899"})
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", font_color="#344054", legend_title_text="", margin=dict(l=10,r=10,t=25,b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(comp.pivot(index="Statistic", columns="Player", values="Mean value").reset_index(), use_container_width=True, hide_index=True)
            else:
                st.info("Select one or more statistics to display a comparison.")
            st.caption("Values shown are dataset averages, not an individual player's career statistics.")

elif page == "Model Evaluation":
    st.markdown('<div class="eyebrow">MATCH INTELLIGENCE / EVALUATION</div>', unsafe_allow_html=True)
    st.title("Model Evaluation")
    st.caption("Review the reported held-out performance of the current model.")
    render_hero("EVALUATION SUMMARY", "Performance, interpreted carefully.", "The current project results indicate modest predictive signal above chance. Metrics should be interpreted alongside the validation design and the dataset's limitations.")
    cols = st.columns(3)
    for col, (label, value) in zip(cols, REPORTED_METRICS.items()):
        with col:
            st.metric(label, f"{value:.2%}")
    chart = pd.DataFrame({"Metric": list(REPORTED_METRICS.keys()), "Score": list(REPORTED_METRICS.values())})
    fig = px.bar(chart, x="Metric", y="Score", text=chart["Score"].map(lambda x: f"{x:.1%}"), color="Metric", color_discrete_sequence=["#2864F0", "#7856FF", "#0EA5A4"])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(range=[0, 0.65], tickformat=".0%", gridcolor="#EDF0F6", title="Score")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font_color="#344054", margin=dict(l=10,r=20,t=25,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Interpretation")
    st.markdown("The held-out test accuracy is **53.02%**, balanced accuracy is **51.99%**, and ROC-AUC is **0.5391**. These results suggest limited predictive power; they should not be presented as strong performance.")
    st.markdown("### Evaluation context")
    st.markdown("These are reported project metrics, not recalculated by this dashboard. For dissertation reporting, describe the split strategy, class distribution, validation design, and any map- or player-grouped tests alongside these values.")

elif page == "About":
    st.markdown('<div class="eyebrow">MATCH INTELLIGENCE / PROJECT NOTES</div>', unsafe_allow_html=True)
    st.title("About This Dashboard")
    st.caption("Model scope, data requirements, and interpretation notes.")
    render_hero("SYSTEM DOCUMENTATION", "An analytical interface for early-game match prediction.", "This application brings together a trained machine learning model and a paired-match dataset to explore early-game StarCraft II statistics, estimate match outcomes, and review model performance.")

    st.markdown("### System overview")
    st.write("The dashboard investigates the predictive value of player statistics available approximately 60 seconds after a match begins. It supports outcome estimation, descriptive player comparisons, and review of evaluation results. It is intended for academic research and exploratory analysis.")

    st.markdown("### Model configuration")
    cfg1, cfg2 = st.columns(2)
    with cfg1:
        st.markdown('<div class="panel"><div class="panel-title">Model and dataset</div><p class="panel-copy">Trained estimator and paired-match data used by the application.</p></div>', unsafe_allow_html=True)
        st.markdown("**Model**")
        st.code("final_sc2_winner_model.joblib", language="text")
        st.markdown("**Dataset**")
        st.code("sc2_paired_training.csv", language="text")
    with cfg2:
        st.markdown('<div class="panel"><div class="panel-title">Prediction specification</div><p class="panel-copy">The target and feature types expected by the model.</p></div>', unsafe_allow_html=True)
        st.markdown("- **Prediction point:** Approximately 60 seconds into the replay")
        st.markdown("- **Target variable:** `player1_win`")
        st.markdown("- **Target interpretation:** `1` means Player 1 wins; `0` means Player 1 loses")
        st.markdown("- **Input features:** Numeric player statistics and race categories when included in the model")

    st.markdown("### Current system status")
    status_cols = st.columns(3)
    with status_cols[0]:
        render_metric_card("Model features", f"{num_features}" if feature_names else "Unavailable", "Detected input columns")
    with status_cols[1]:
        render_metric_card("Dataset rows", f"{len(data):,}" if data is not None else "Unavailable", "Rows in loaded dataset")
    with status_cols[2]:
        render_metric_card("Model status", "Loaded" if model is not None else "Unavailable", "Current application session")
    with st.expander("View file paths and connection details"):
        st.markdown("**Model path**")
        st.code(str(resolve_path(model_path)), language="text")
        st.markdown("**Dataset path**")
        st.code(str(resolve_path(data_path)), language="text")
        if model is not None:
            st.success("The trained model is loaded successfully.")
        else:
            st.error(model_error or "The trained model is unavailable.")
        if data is not None:
            st.success("The paired-match dataset is loaded successfully.")
        else:
            st.warning(data_error or "The paired-match dataset is unavailable.")

    st.markdown("### Methodology and interpretation")
    st.write("The model estimates the probability of a Player 1 victory from the supplied early-game features. The estimate reflects patterns learned from the training data and should be interpreted alongside the model's measured evaluation results.")
    st.markdown("### Limitations and intended use")
    st.write("Predictions cannot account for every strategic decision, hidden information, player adaptation, map-specific conditions, or events that occur after the selected prediction point. Performance may also be affected by data quality and differences between the training data and new matches. Outputs are exploratory estimates, not guarantees of match outcomes. This dashboard complements the experimental analysis documented in the dissertation; it does not replace rigorous validation or critical interpretation.")

# Consistent footer
st.markdown("---")
st.markdown("<p class='small-note'>STARCRAFT II MATCH INTELLIGENCE · Early-game prediction research · Estimates are not guarantees.</p>", unsafe_allow_html=True)
