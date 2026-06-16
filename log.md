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

## 2026-06-16 — GridSearchCV (Max R) + Cross-Entropy Optimization

- **Max R — GridSearchCV hyperparameter tuning**:
  - Added `train_with_gridsearch()` to `modeling.py` — exhaustive search over all hyperparameter combinations
  - Parameter grids: LR (20 combos), KNN (20), SVC (16), RF (36), DT (30) = **122 total combinations** tested
  - Each combo evaluated with 5-fold CV; best combination selected by max CV accuracy
  - Best results: SVC CV=98.33% (`C=1, gamma=0.1`), LR test=100% (`C=10, max_iter=500`)
  - Phase 4 displays full tuning table per model — all parameter combinations sorted by CV score
- **Cross-Entropy (Log Loss) — Process & Results**:
  - Dedicated section in Phase 4 with LaTeX formula explanation
  - Log Loss bar chart across all 5 models (horizontal, green=best, red=worst)
  - Log Loss values: LR 0.0833 (best, cross-entropy trained), KNN 0.1196, RF 0.1317, SVC 0.1657, DT 2.4029
  - Explanation of `solver='lbfgs'` minimizing multinomial cross-entropy during training
- **R² Score** added to `compute_all_metrics()` and home dashboard (7 st.metric cards now)
- **Cache fix**: `cv_results_df` KeyError from stale Streamlit Cloud cache — `info.get("cv_results_df")` with warning fallback

## 2026-06-16 — Log Files

- Created `log.md` with chronological development history (HW6 style)
- Updated `README.md` with GridSearchCV results, 3D PCA chart, Cross-Entropy model performance table
- Final git history: 12 commits on `main` branch
