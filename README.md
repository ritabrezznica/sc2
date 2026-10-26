# StarCraft II Early-Game Match Prediction

A machine-learning project that explores whether StarCraft II match outcomes can be predicted from player statistics collected during the early game.

The project includes a Streamlit dashboard for exploring match statistics, comparing players, generating winner-probability estimates, and reviewing model evaluation results.

> ## Table of Contents

- [Project Overview](#project-overview)
- [Research Objective](#research-objective)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Dashboard Features](#dashboard-features)
- [Project Files](#project-files)
- [Installation](#installation)
- [Run the Dashboard](#run-the-dashboard)
- [Reproducibility and Validation](#reproducibility-and-validation)
- [Limitations](#limitations)
- [Future Work](#future-work)

## Project Overview

StarCraft II is a real-time strategy game in which match outcomes depend on multiple interacting factors, including resource management, army development, technology, and player decisions.

This project investigates how much information early-game player statistics provide about the eventual match winner.

The dashboard uses a saved machine-learning pipeline and a paired-match dataset to support interactive exploration and prediction.

## Research Objective

The main research question is:

**How effectively can the eventual winner of a StarCraft II match be predicted using statistics available during approximately the first 60 seconds of gameplay?**

The project also investigates whether model performance changes when evaluation is performed on maps or players excluded from training.

## Dataset

The current paired training dataset contains **23,423 matches** and **82 columns**.

The dataset includes a replay index and paired player-level features. The precise feature definitions and target-label construction should be consulted in the project's data-processing and training scripts.

The application loads the following files by default:

- `sc2_paired_training.csv` — paired-match dataset.
- `final_sc2_winner_model.joblib` — saved prediction pipeline.

The application uses these resources to support its dashboard and prediction workflow.

## Model Evaluation

The project evaluates model performance using accuracy, balanced accuracy, and ROC AUC. Additional experiments investigate generalisation to unseen maps and unseen players.

### Reported results

| Evaluation | Accuracy | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|
| Saved model, held-out test | 53.02% | 51.99% | 0.5391 |
| Logistic Regression, 5-fold CV | 53.31% ± 0.81 percentage points | 52.08% ± 0.79 percentage points | 0.5371 ± 0.0098 |
| XGBoost, 5-fold CV | 52.63% ± 0.30 percentage points | 51.67% ± 0.32 percentage points | 0.5307 ± 0.0061 |
| Unseen-map evaluation, statistics + race + matchup | 53.39% ± 0.89 percentage points | 52.47% ± 0.79 percentage points | 0.5402 ± 0.0076 |

The cross-validation results are reported as mean ± standard deviation across folds. The saved model's held-out test results are reported separately and should not be directly treated as the same experiment.

### Interpretation

The results indicate that early-game statistics contain some predictive signal, but the measured performance is limited.

Accuracy near 53% represents only a small improvement over a balanced 50% chance baseline. ROC AUC values near 0.54 also indicate weak discrimination between the two outcomes.

Performance on unseen maps is important because a model may otherwise learn patterns associated with maps represented in both training and evaluation data.

These findings do not establish that the model will perform equally well in live matches, across different player populations, or under different game conditions.

## Dashboard Features

The Streamlit application provides the following sections:

- **Overview** — summary statistics and project information.
- **Match Prediction** — estimate the probability of a Player 1 victory using the available prediction workflow.
- **Player Comparison** — explore paired player statistics.
- **Model Evaluation** — review evaluation metrics and visualisations.
- **About** — project context and explanatory information.

Predictions should be interpreted as model estimates rather than certain outcomes.

## Project Files

The main files required to launch the dashboard are:

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard and application logic |
| `requirements.txt` | Pinned Python package dependencies |
| `sc2_paired_training.csv` | Paired-match dataset (82 columns, including race) used by the app |
| `final_sc2_winner_model.joblib` | Saved machine-learning pipeline |
| `src/extract/` | Replay extraction, cleaning and dataset-building scripts |
| `src/train/` | Model training experiments |
| `src/evaluate/` | Evaluation and validation scripts (unseen maps and players, SHAP) |
| `results/` | Experiment output CSVs (model comparisons, unseen-map and unseen-player results) |
| `legacy/` | Earlier scripts for a different pipeline, kept for traceability |
| `models/` | Additional model artifacts, where included |

Run scripts from the repository root, for example `python src/train/train_final_model.py`, so relative file paths resolve. Scripts write their output files to the root; move them into `results/` if you want to keep them organised.

Some diagnostic scripts and intermediate files are retained for research traceability. Not every script is required to run the dashboard.

Large raw replay archives and intermediate datasets are excluded from the intended public repository to keep it manageable. Reproducing the full data pipeline may therefore require separately obtaining the original data and configuring the relevant input paths.

## Installation

### Requirements

- Python with the required packages installed.
- The project files listed above.
- The saved model and paired dataset in the project directory.

The dashboard has been checked in a Python 3.13.9 environment.

### 1. Clone the repository

```bash
git clone https://github.com/ritabrezznica/sc2.git
cd sc2
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can use the environment's Python executable directly instead.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Check the required files

Confirm that these files exist in the project directory:

```text
app.py
requirements.txt
sc2_paired_training.csv
final_sc2_winner_model.joblib
```

## Run the Dashboard

Start the application with:

```bash
python -m streamlit run app.py
```

Streamlit normally makes the dashboard available at:

```text
http://localhost:8501
```

To stop the application, press `Ctrl+C` in the terminal.

## Reproducibility and Validation

Before running the dashboard, you can check Python syntax:

```bash
python -m py_compile app.py
```

The saved model and dataset can also be tested for basic loading:

```python
import joblib
import pandas as pd

model = joblib.load("final_sc2_winner_model.joblib")
data = pd.read_csv("sc2_paired_training.csv", nrows=5)

print("Model:", type(model).__name__)
print("Sample rows:", len(data))
print("Columns:", len(data.columns))
```

The project contains additional scripts for data preparation, model comparisons, validation, and experiments involving unseen maps and players. Some scripts depend on raw replay data or intermediate files that are not included in the public repository.

For reproducible research, record the dataset version, feature definitions, train/test split strategy, random seeds, dependency versions, and model artifact used for each reported result.

## Limitations

- **Limited predictive performance:** the reported results are only slightly above chance.
- **Generalisation:** results on held-out matches do not guarantee performance on unseen players, maps, or future game versions.
- **Data dependencies:** the full replay-processing pipeline may require data excluded from the repository.
- **Model and data compatibility:** the saved model expects input features in a compatible schema and order. Arbitrary CSV files may not work.
- **Probabilities are estimates:** model outputs do not guarantee a match result.
- **Evaluation context matters:** metrics from different splits or experiments should not be compared as if they came from one identical test.

## Future Work

Potential improvements include:

- Further investigating generalisation to unseen players and maps.
- Examining feature quality and possible data leakage.
- Comparing simple baselines with more complex models.
- Improving the reproducibility of data extraction and model evaluation.
- Testing whether additional early-game information improves performance.
- Documenting feature definitions and data provenance more fully.

## Research Context

This repository supports an academic project on StarCraft II early-game statistics and machine-learning-based match outcome prediction.

The goal is to investigate the predictive value of early-game information, report evaluation results transparently, and provide an interactive interface for exploring the findings.
