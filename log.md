# Work Log — Iris Species Classification (CRISP-DM)

## 2026-06-16 — Project Initialization

- Created CRISP-DM project architecture following `design.md` blueprint
- Set up directory tree: `src/`, `models/`, `data/raw/`, `data/processed/`
- Initial commit: `design.md`, `app.py`, `requirements.txt`, 4 source modules
  - `src/data_understanding.py` — Phase 2: READ-ONLY EDA, PCA (2D), explained variance chart
  - `src/data_preparation.py` — Phase 3: consolidated preprocessing (StratifiedSplit + StandardScaler)
  - `src/modeling.py` — Phase 4: 5 classifiers (LR, KNN, SVC, RF, DT) + cross-validation
  - `src/evaluation.py` — Phase 5: accuracy table, confusion matrices, ROC curves
- **sklearn 1.9.0 compatibility fixes**:
  - Removed deprecated `multi_class` param from `LogisticRegression`; cross-entropy loss achieved via `solver='lbfgs'`
  - Replaced `SVC(probability=True)` with `CalibratedClassifierCV(SVC(), ensemble=False)` per 1.9+ deprecation
- Streamlit app with 7 CRISP-DM phases in sidebar radio navigation
- Push to GitHub: https://github.com/miccowang66-max/New-Model-Iris-Classification

## 2026-06-16 — README + Performance Metrics

- Created `README.md` following HW6 style: badges, live demo table, CRISP-DM workflow table, features, project tree, tech stack, deployment
- **Performance Metrics Dashboard** — new Phase 6 in app:
  - Added `compute_all_metrics()` to `src/evaluation.py`: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), F1 (Macro/Weighted), Log Loss, Matthews Correlation Coefficient, Cohen's Kappa
  - `st.metric` cards for 10 metrics per selected model
  - Full comparison table with color-graded cells across all 5 models
  - Side-by-side bar chart: Accuracy / F1 / Precision / Recall
  - Dynamic confusion matrix tied to model selector
- Added Streamlit Cloud deploy badge to README
- Re-numbered sidebar phases: 1→7

## 2026-06-16 — Streamlit Cloud Fixes

- **SyntaxError fix**: Curly apostrophe `\u2019` (RIGHT SINGLE QUOTATION MARK) in `"Cohen\u2019s Kappa"` key broke f-string parsing on Streamlit Cloud (Python 3.12+). Replaced with ASCII `'`.
- **Prediction not working**: `.gitignore` excluded `*.pkl` — model files (`best_model.pkl`, `scaler.pkl`) never deployed to Streamlit Cloud.
  - Root cause: `render_phase_6` loaded model from disk via `joblib.load()`, but pkl files absent on server
  - Solution: Removed `*.pkl` from `.gitignore`, committed pre-trained SVC model (96.67% test accuracy)
  - Added in-memory model fallback for robustness
- Updated live demo URL to https://new-model-iris-classification.streamlit.app

## 2026-06-16 — PCA Charts & Home Dashboard

- **PCA explained variance chart** requirement — user reference image `sphx_glr_plot_pca_iris_002.png`
  - Saved static PNG to repo with matching filename
  - Added `🏠 Home — Key Results` as default sidebar option
  - Home dashboard shows PCA charts + model performance at load time
  - Embedded chart in README Key Results section
- **3D PCA projection**:
  - Added `plot_pca_3d()` to `data_understanding.py` using `mpl_toolkits.mplot3d`
  - 3D scatter: PC1, PC2, PC3 with species colors + marker shapes (circles, triangles, squares), depth shading, `view_init(25, 45)`
  - Saved static PNG (`pca_3d_projection.png`) to repo + README
  - Displayed in Home dashboard and Phase 2 PCA tab

## 2026-06-16 — Layout & Polish

- Swapped Home dashboard layout: PCA explained variance (left) + 3D projection (right)
- Removed `*.png` and `*.jpg` from `.gitignore` to allow static chart files in repo
- Final git history: 8 commits on `main` branch
