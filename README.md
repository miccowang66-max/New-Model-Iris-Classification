# Iris Species Classification — CRISP-DM ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4c72b0?style=flat-square)](https://seaborn.pydata.org/)
[![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-green?style=flat-square)](https://www.datascience-pm.com/crisp-dm-2/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](./LICENSE)

> A production-ready, interactive Streamlit dashboard for classifying Iris flowers using 5 machine learning models under the **CRISP-DM** methodology. Features **PCA explained variance charts**, **cross-validation**, **ROC curves**, and **live prediction**.

---

## 🖥️ Live Demo

| Platform | URL | Description |
|----------|-----|-------------|
| **Streamlit Cloud** | [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=miccowang66-max/New-Model-Iris-Classification&branch=main&mainModule=app.py) | Production deployment |
| **Local** | `streamlit run app.py` | Local development server |

> Click the Streamlit badge above to deploy instantly, or run locally with the command below.

---

## 📊 CRISP-DM Workflow

| Phase | Name | Module | Key Deliverables |
|-------|------|--------|------------------|
| 1 | Business Understanding | `app.py` | Problem definition, success criteria (Accuracy > 90%) |
| 2 | Data Understanding | `src/data_understanding.py` | READ-ONLY EDA, Pairplot, Correlation Heatmap, **PCA Explained Variance Chart** |
| 3 | Data Preparation | `src/data_preparation.py` | 80/20 Stratified Split, `StandardScaler` (fit on train only), no data leakage |
| 4 | Modeling | `src/modeling.py` | 5 classifiers: Logistic Regression (Cross-Entropy), KNN, SVC, Random Forest, Decision Tree |
| 5 | Evaluation | `src/evaluation.py` | 5-Fold CV, Test Accuracy, Confusion Matrices, ROC/AUC, Classification Reports |
| 6 | Deployment | `app.py` | Streamlit interactive prediction with confidence scores |

---

## ✨ Features

### Data Understanding (READ-ONLY Phase)
- 🔍 **Dataset Overview** — Shape, dtypes, null counts, descriptive statistics, class distribution
- 🏷 **Visual EDA** — Seaborn pairplot with species hue, correlation heatmap
- 🧬 **PCA Analysis** — 2D PCA projection scatter plot
- 📊 **Explained Variance Chart** — Bar chart matching scikit-learn PCA iris reference (individual + cumulative)

### Modeling & Evaluation
- ⚡ **5 Classifiers Compared** — Logistic Regression (Cross-Entropy loss via `lbfgs`), KNN (k=5), SVC (RBF kernel), Random Forest (100 trees), Decision Tree (max_depth=5)
- 🔁 **5-Fold Cross-Validation** — `cross_val_score` with mean ± std for every model
- 📈 **Model Comparison Dashboard** — Horizontal bar charts for test accuracy + CV accuracy with error bars
- 🎯 **Confusion Matrices** — Heatmaps for all 5 models side-by-side
- 📉 **ROC Curves** — One-vs-Rest ROC + AUC per species per model
- 📋 **Classification Reports** — Precision / Recall / F1 per class

### Performance Metrics Dashboard
- 📊 **10 Comprehensive Metrics** — Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 Score (Macro/Weighted), Log Loss, Matthews Correlation Coefficient, Cohen's Kappa
- 🏆 **Per-Model Metric Cards** — `st.metric` dashboard with top-row summary for any selected model
- 📋 **Full Comparison Table** — All 10 metrics across all 5 models with color-graded cells
- 📊 **Metric Bar Chart** — Side-by-side Accuracy / F1 / Precision / Recall per model
- 🎯 **Selective Confusion Matrix** — Dynamic heatmap for the selected model only

### Live Prediction (Phase 6)
- 🎚 **Interactive Sliders** — Sepal length/width, Petal length/width
- 🌸 **Real-Time Species Classification** — Instant prediction with emoji result card
- 📊 **Confidence Distribution** — Bar chart showing probability per species
- 💡 **Reference Measurements** — Typical values table for each species

### Architecture
- 🏗 **CRISP-DM Design First** — Full `design.md` blueprint before any code
- 🔒 **Zero Data Leakage** — `StandardScaler` fitted on `X_train` only; `stratify=y` for balanced split
- 📦 **Consolidated Preprocessing** — All transforms in single `data_preparation.py` module
- ⛔ **READ-ONLY Enforcement** — Phase 2 strictly view-only, no mutations

---

## 📁 Project Structure

```
Iris_Classification/
├── .gitignore
├── design.md                         # Single source of truth — architecture blueprint
├── requirements.txt                  # Pinned Python dependencies
├── README.md                         # ← This file
│
├── data/
│   ├── raw/                          # READ-ONLY (Iris loaded from sklearn on first run)
│   └── processed/                    # Generated by pipeline
│       ├── X_train.csv               # Scaled features (80%)
│       ├── X_test.csv                # Scaled features (20%)
│       ├── y_train.csv               # Target labels (80%)
│       └── y_test.csv                # Target labels (20%)
│
├── src/
│   ├── __init__.py
│   ├── data_understanding.py         # Phase 2: EDA, PCA, explained variance chart
│   ├── data_preparation.py           # Phase 3: Split, Scale, Persist (CONSOLIDATED)
│   ├── modeling.py                   # Phase 4: 5-model training + cross-validation
│   └── evaluation.py                 # Phase 5: Metrics, confusion matrices, ROC, CV comparison
│
├── models/
│   ├── best_model.pkl                # Auto-saved best model by test accuracy
│   └── scaler.pkl                    # Serialized StandardScaler for inference
│
├── notebooks/
│   └── crisp_dm_exploration.ipynb    # Optional ad-hoc exploration
│
└── app.py                            # Streamlit deployment (Phase 6) — interactive dashboard
```

---

## 🛠 Tech Stack

| Technology | Purpose | Version |
|------------|---------|---------|
| [Python](https://www.python.org) | Core language | >= 3.9 |
| [pandas](https://pandas.pydata.org) | Data manipulation | >= 1.5 |
| [numpy](https://numpy.org) | Numerical engine | >= 1.24 |
| [scikit-learn](https://scikit-learn.org) | Models, preprocessing, metrics, PCA | >= 1.9 |
| [matplotlib](https://matplotlib.org) | Static charts | >= 3.7 |
| [seaborn](https://seaborn.pydata.org) | Statistical visualizations | >= 0.12 |
| [streamlit](https://streamlit.io) | Web deployment dashboard | >= 1.28 |
| [joblib](https://joblib.readthedocs.io) | Model serialization | >= 1.3 |

---

## 🔧 Local Development

```bash
# Clone the repository
git clone https://github.com/miccowang66-max/New-Model-Iris-Classification.git
cd New-Model-Iris-Classification

# (Optional) Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit dashboard
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> Python 3.9+ is required.

---

## 🚀 Deployment

### Streamlit Community Cloud (Recommended)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select repo → Set `app.py` as main file path
4. Click **"Deploy"** — Streamlit auto-detects `requirements.txt`

### Run as Standalone Script

```bash
# Train models and run evaluation from CLI
python -c "
from src.data_understanding import load_iris_data, plot_explained_variance
from src.data_preparation import prepare_data
from src.modeling import train_all_models, save_best_model
from src.evaluation import evaluate_with_cv

df = load_iris_data()
X_train, X_test, y_train, y_test, _ = prepare_data(df)
models = train_all_models(X_train, y_train)
best, score = save_best_model(models, X_test, y_test)
results = evaluate_with_cv(models, X_train, y_train, X_test, y_test)
print(results.to_string(index=False))
print(f'Best: {best} ({score:.2%})')
"
```

---

## 📄 Commands

| Command | Description |
|---------|-------------|
| `streamlit run app.py` | Launch interactive Streamlit dashboard |
| `python -m pytest` | Run test suite (coming soon) |
| `pip install -r requirements.txt` | Install all dependencies |

---

## 📄 License

MIT
