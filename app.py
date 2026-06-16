import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_understanding import (
    load_iris_data, display_shape, display_describe, display_null_counts,
    display_head, display_class_distribution, plot_class_distribution,
    plot_pairplot, plot_correlation_heatmap, plot_pca_2d, plot_explained_variance
)
from src.data_preparation import prepare_data
from src.modeling import train_all_models, save_best_model
from src.evaluation import (
    evaluate_all, evaluate_with_cv, get_classification_reports,
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison, plot_cv_comparison,
    compute_all_models_metrics
)

st.set_page_config(
    page_title="Iris Classification — CRISP-DM",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_STYLE = """
<style>
    .main-header { font-size: 2.2em; font-weight: 700; color: #1f77b4; margin-bottom: 0; }
    .sub-header { font-size: 1.1em; color: #666; margin-top: -10px; margin-bottom: 20px; }
    .phase-title { font-size: 1.5em; font-weight: 600; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 5px; }
    .metric-box { background: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .metric-value { font-size: 2em; font-weight: 700; color: #1f77b4; }
    .metric-label { font-size: 0.85em; color: #666; }
    .prediction-result { background: #e8f5e9; border-radius: 10px; padding: 20px; text-align: center; margin-top: 15px; }
    .prediction-species { font-size: 2.5em; font-weight: 700; color: #2e7d32; }
</style>
"""
st.markdown(PAGE_STYLE, unsafe_allow_html=True)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@st.cache_data
def cached_load_data():
    return load_iris_data()


@st.cache_resource
def cached_train_models(_X_train, _y_train):
    return train_all_models(_X_train, _y_train)


@st.cache_data
def cached_prepare_data(_df):
    return prepare_data(_df, save=True)


def load_or_train():
    df = cached_load_data()
    X_train, X_test, y_train, y_test, scaler = cached_prepare_data(df)
    models = cached_train_models(X_train, y_train)

    if not os.path.exists(BEST_MODEL_PATH):
        save_best_model(models, X_test, y_test)

    return df, X_train, X_test, y_train, y_test, scaler, models


def main():
    st.markdown('<p class="main-header">🌸 Iris Species Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">CRISP-DM Machine Learning Pipeline · PCA + Multi-Model Evaluation · Streamlit Deployment</p>', unsafe_allow_html=True)

    st.sidebar.markdown("## 📋 CRISP-DM Phases")
    phase = st.sidebar.radio(
        "Navigate:",
        [
            "1. Business Understanding",
            "2. Data Understanding",
            "3. Data Preparation",
            "4. Modeling",
            "5. Evaluation",
            "6. Performance Metrics",
            "7. Deployment — Predict",
        ],
    )

    with st.spinner("Loading Iris dataset & training models..."):
        df, X_train, X_test, y_train, y_test, scaler, models = load_or_train()

    if phase == "1. Business Understanding":
        render_phase_1()

    elif phase == "2. Data Understanding":
        render_phase_2(df)

    elif phase == "3. Data Preparation":
        render_phase_3(df, X_train, X_test, y_train, y_test)

    elif phase == "4. Modeling":
        render_phase_4(models)

    elif phase == "5. Evaluation":
        render_phase_5(models, X_train, y_train, X_test, y_test)

    elif phase == "6. Performance Metrics":
        render_performance_metrics(models, X_test, y_test)

    elif phase == "7. Deployment — Predict":
        render_phase_6(scaler)


def render_phase_1():
    st.markdown('<p class="phase-title">Phase 1 — Business Understanding</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### 🎯 Project Objective
        Build a machine learning model to **classify Iris flowers** into one of three species
        — *setosa*, *versicolor*, or *virginica* — using four morphological measurements.

        ### 📊 Dataset
        The **Iris dataset** (Fisher, 1936) contains 150 samples, 50 from each of 3 species:
        | # | Feature | Description |
        |---|---------|-------------|
        | 1 | Sepal Length | Length of the sepal (cm) |
        | 2 | Sepal Width | Width of the sepal (cm) |
        | 3 | Petal Length | Length of the petal (cm) |
        | 4 | Petal Width | Width of the petal (cm) |
        """)

    with col2:
        st.markdown("""
        ### ✅ Success Criteria
        - **Accuracy > 90%** on held-out test set
        - Compare **5 classification models**
        - Deploy best model via **Streamlit**
        - Final report with **PCA explained variance chart**

        ### 🧭 CRISP-DM Methodology
        - Phase 2: Data Understanding
        - Phase 3: Data Preparation
        - Phase 4: Modeling
        - Phase 5: Evaluation
        - Phase 6: Deployment
        """)

    st.info("Use the **sidebar menu** to navigate through each CRISP-DM phase.")


def render_phase_2(df):
    st.markdown('<p class="phase-title">Phase 2 — Data Understanding (READ-ONLY)</p>', unsafe_allow_html=True)
    st.markdown("⛔ **This phase is strictly read-only.** No transformations, drops, or writes are performed.")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Distributions", "🔗 Correlations", "🧬 PCA Analysis"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        shape = display_shape(df)
        nulls = display_null_counts(df)
        with col1:
            st.markdown('<div class="metric-box"><div class="metric-value">{}</div><div class="metric-label">Samples (rows)</div></div>'.format(shape[0]), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-box"><div class="metric-value">{}</div><div class="metric-label">Features (cols)</div></div>'.format(shape[1]), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-box"><div class="metric-value">{}</div><div class="metric-label">Missing Values</div></div>'.format(nulls.sum()), unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-box"><div class="metric-value">3</div><div class="metric-label">Species Classes</div></div>'.format(), unsafe_allow_html=True)

        st.markdown("#### Data Sample (head)")
        st.dataframe(display_head(df, 10), use_container_width=True)

        st.markdown("#### Descriptive Statistics")
        st.dataframe(display_describe(df), use_container_width=True)

        st.markdown("#### Class Distribution")
        st.dataframe(display_class_distribution(df).to_frame("Count"), use_container_width=True)

    with tab2:
        st.markdown("#### Class Distribution Bar Chart")
        st.pyplot(plot_class_distribution(df))

        st.markdown("#### Pairplot — Feature Relationships by Species")
        with st.spinner("Rendering pairplot..."):
            st.pyplot(plot_pairplot(df))

    with tab3:
        st.markdown("#### Feature Correlation Heatmap")
        st.pyplot(plot_correlation_heatmap(df))

        st.markdown("""
        **Observations:**
        - Petal length and petal width are strongly positively correlated (~0.96)
        - Sepal length is moderately correlated with petal dimensions
        - Sepal width has weak negative correlation with other features
        """)

    with tab4:
        st.markdown("#### PCA — 2D Dimensionality Reduction")
        st.pyplot(plot_pca_2d(df))

        st.markdown("---")
        st.markdown("#### PCA — Explained Variance Ratio")
        st.markdown("*This chart matches the scikit-learn PCA Iris explained variance reference plot.*")
        st.pyplot(plot_explained_variance())

        st.markdown("""
        **PCA Insights:**
        - PC1 captures ~73% of variance — dominated by petal features
        - PC2 captures ~23% — dominated by sepal features
        - Together PC1+PC2 explain ~96% of total variance
        - Classes are well-separated in PCA space, confirming the problem is highly solvable
        """)


def render_phase_3(df, X_train, X_test, y_train, y_test):
    st.markdown('<p class="phase-title">Phase 3 — Data Preparation</p>', unsafe_allow_html=True)
    st.markdown("All preprocessing is **consolidated in `data_preparation.py`** per the project architecture.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Preprocessing Steps")
        st.markdown("""
        | Step | Operation | Details |
        |------|-----------|---------|
        | 1 | Define X, y | Features = 4 measurements, Target = species |
        | 2 | Train/Test Split | 80/20 split, `stratify=y`, `random_state=42` |
        | 3 | Feature Scaling | `StandardScaler` — fit on train only |
        | 4 | Persist | Save splits to `data/processed/` |
        | 5 | Serialize | Save scaler to `models/scaler.pkl` |
        """)

        st.warning("""
        ⚠️ **Data Leakage Prevention:**
        `StandardScaler` is fitted **only** on `X_train`, then used to transform both
        `X_train` and `X_test`. This ensures no information from the test set leaks
        into model training.
        """)

    with col2:
        st.markdown("### Split Summary")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)

        split_df = pd.DataFrame({
            "Set": ["Training (80%)"] * 3 + ["Test (20%)"] * 3,
            "Species": ["setosa", "versicolor", "virginica"] * 2,
            "Count": [counts_train[0], counts_train[1], counts_train[2],
                       counts_test[0], counts_test[1], counts_test[2]],
        })
        st.dataframe(split_df, use_container_width=True)

        st.markdown("### Scaled Data Preview (Train)")
        preview = pd.DataFrame(
            X_train[:5],
            columns=["sepal_length (scaled)", "sepal_width (scaled)", "petal_length (scaled)", "petal_width (scaled)"]
        ).round(4)
        st.dataframe(preview, use_container_width=True)


def render_phase_4(models):
    st.markdown('<p class="phase-title">Phase 4 — Modeling</p>', unsafe_allow_html=True)

    st.markdown("""
    Five classification models are trained with `random_state=42` for reproducibility.
    All models use the **same** standardized training data from Phase 3.
    """)

    cols = st.columns(len(models))
    for idx, (name, model) in enumerate(models.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            params = model.get_params()
            key_params = {}
            if "C" in params:
                key_params["C (regularization)"] = params["C"]
            if "n_neighbors" in params:
                key_params["n_neighbors"] = params["n_neighbors"]
            if "kernel" in params:
                key_params["kernel"] = params["kernel"]
            if "n_estimators" in params:
                key_params["n_estimators"] = params["n_estimators"]
            if "max_depth" in params:
                key_params["max_depth"] = params["max_depth"]
            if "max_iter" in params:
                key_params["max_iter"] = params["max_iter"]

            param_str = "\n".join([f"- {k}: `{v}`" for k, v in key_params.items()])
            if param_str:
                st.markdown(param_str)

    st.info("Go to **Evaluation** phase to see model performance comparison.")


def render_phase_5(models, X_train, y_train, X_test, y_test):
    st.markdown('<p class="phase-title">Phase 5 \u2014 Evaluation</p>', unsafe_allow_html=True)

    results = evaluate_with_cv(models, X_train, y_train, X_test, y_test, cv=5)

    st.markdown("### 5-Fold Cross-Validation Accuracy Comparison")
    st.pyplot(plot_cv_comparison(results))

    st.markdown("### Test Accuracy Comparison")
    st.pyplot(plot_model_comparison(results))

    st.markdown("### Full Evaluation Table (Cross-Validation + Test)")

    display_cols = results[["Model", "CV Mean Accuracy", "CV Std", "Test Accuracy"]].copy()
    display_cols["CV Mean Accuracy"] = display_cols["CV Mean Accuracy"].apply(lambda x: f"{x:.4f}")
    display_cols["CV Std"] = display_cols["CV Std"].apply(lambda x: f"{x:.4f}")
    display_cols["Test Accuracy"] = display_cols["Test Accuracy"].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_cols, use_container_width=True)

    best_model_name = results.iloc[0]["Model"]
    best_test_acc = results.iloc[0]["Test Accuracy"]
    best_cv_mean = results.iloc[0]["CV Mean Accuracy"]
    st.success(f"Best Model: **{best_model_name}** | Test Acc: {best_test_acc:.2%} | 5-Fold CV Mean: {best_cv_mean:.2%}")

    st.markdown("---")
    st.markdown("### Classification Reports")
    reports = get_classification_reports(models, X_test, y_test)
    selected_report = st.selectbox("Select model to view report:", list(models.keys()))
    if selected_report:
        report = reports[selected_report]
        df_report = pd.DataFrame(report).T.round(3)
        st.dataframe(df_report, use_container_width=True)

    st.markdown("---")
    st.markdown("### Confusion Matrices")
    st.pyplot(plot_confusion_matrices(models, X_test, y_test))

    st.markdown("---")
    st.markdown("### ROC Curves (One-vs-Rest)")
    st.pyplot(plot_roc_curves(models, X_test, y_test))


def render_performance_metrics(models, X_test, y_test):
    st.markdown('<p class="phase-title">Performance Metrics \u2014 All Models</p>', unsafe_allow_html=True)

    all_metrics = compute_all_models_metrics(models, X_test, y_test)

    selected_model = st.selectbox(
        "Select model to view detailed metrics:",
        list(models.keys()),
        index=list(models.keys()).index(
            max(all_metrics, key=lambda m: all_metrics[m]["Accuracy"])
        ),
    )

    metrics = all_metrics[selected_model]
    y_pred = models[selected_model].predict(X_test)
    y_proba = None
    if hasattr(models[selected_model], "predict_proba"):
        y_proba = models[selected_model].predict_proba(X_test)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
    with col2:
        st.metric("F1 (Macro)", f"{metrics['F1 Score (Macro)']:.2%}")
    with col3:
        st.metric("Precision (Macro)", f"{metrics['Precision (Macro)']:.2%}")
    with col4:
        st.metric("Recall (Macro)", f"{metrics['Recall (Macro)']:.2%}")
    with col5:
        ll = metrics.get("Log Loss")
        st.metric("Log Loss", f"{ll:.4f}" if ll is not None else "N/A")

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.metric("F1 (Weighted)", f"{metrics['F1 Score (Weighted)']:.2%}")
    with col7:
        st.metric("Precision (Wtd)", f"{metrics['Precision (Weighted)']:.2%}")
    with col8:
        st.metric("Recall (Wtd)", f"{metrics['Recall (Weighted)']:.2%}")
    with col9:
        st.metric("Matthews CC", f"{metrics['Matthews CorrCoef']:.4f}")
    with col10:
        st.metric("Cohen's Kappa", f"{metrics[\"Cohen\u2019s Kappa\"]:.4f}")

    st.markdown("---")
    st.markdown("### Model Comparison Table")
    comparison_rows = []
    for name, m in all_metrics.items():
        row = {
            "Model": name,
            "Accuracy": m["Accuracy"],
            "F1 Macro": m["F1 Score (Macro)"],
            "Precision Macro": m["Precision (Macro)"],
            "Recall Macro": m["Recall (Macro)"],
            "MCC": m["Matthews CorrCoef"],
            "Kappa": m["Cohen\u2019s Kappa"],
        }
        if m.get("Log Loss") is not None:
            row["Log Loss"] = m["Log Loss"]
        comparison_rows.append(row)

    df_compare = pd.DataFrame(comparison_rows)
    df_compare = df_compare.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    format_dict = {col: "{:.4f}" for col in df_compare.columns if col != "Model"}
    st.dataframe(df_compare.style.format(format_dict).background_gradient(
        subset=[c for c in df_compare.columns if c != "Model"],
        cmap="RdYlGn", axis=0
    ), use_container_width=True)

    st.markdown("---")
    st.markdown("### Per-Class Metrics (Macro-Averaged)")

    model_names = list(all_metrics.keys())
    chart_data = pd.DataFrame({
        "Model": model_names,
        "Accuracy": [all_metrics[n]["Accuracy"] for n in model_names],
        "F1 Score": [all_metrics[n]["F1 Score (Macro)"] for n in model_names],
        "Precision": [all_metrics[n]["Precision (Macro)"] for n in model_names],
        "Recall": [all_metrics[n]["Recall (Macro)"] for n in model_names],
    }).set_index("Model")

    st.bar_chart(chart_data, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### Confusion Matrix — {selected_model}")
    y_pred_sel = models[selected_model].predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_data = sklearn_cm(y_test, y_pred_sel)
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues",
                xticklabels=["setosa", "versicolor", "virginica"],
                yticklabels=["setosa", "versicolor", "virginica"],
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(selected_model, fontweight="bold")
    st.pyplot(fig)


def render_phase_6(scaler):
    st.markdown('<p class="phase-title">Phase 6 — Deployment — Live Prediction</p>', unsafe_allow_html=True)

    st.markdown("Enter flower measurements below to predict the Iris species in real time.")

    best_model = None
    if os.path.exists(BEST_MODEL_PATH):
        best_model = joblib.load(BEST_MODEL_PATH)
    else:
        st.error("No trained model found. Please ensure training has completed.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📏 Input Measurements")
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1, format="%.1f")
        sepal_width = st.slider("Sepal Width (cm)", 1.5, 5.0, 3.0, 0.1, format="%.1f")
        petal_length = st.slider("Petal Length (cm)", 0.5, 7.0, 4.0, 0.1, format="%.1f")
        petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 1.3, 0.1, format="%.1f")

        predict_btn = st.button("🔍 Predict Species", type="primary", use_container_width=True)

        st.markdown("""
        ---
        ### 💡 Typical Measurements
        | Species | Sepal L | Sepal W | Petal L | Petal W |
        |---------|---------|---------|---------|---------|
        | Setosa | ~5.0 | ~3.4 | ~1.5 | ~0.2 |
        | Versicolor | ~5.9 | ~2.8 | ~4.3 | ~1.3 |
        | Virginica | ~6.6 | ~3.0 | ~5.5 | ~2.0 |
        """)

    with col2:
        st.markdown("### 🎯 Prediction Result")
        if predict_btn:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_data)

            prediction = best_model.predict(input_scaled)[0]
            species_name = TARGET_NAMES[prediction]

            emoji_map = {"setosa": "🌿", "versicolor": "🌸", "virginica": "🌺"}

            if hasattr(best_model, "predict_proba"):
                proba = best_model.predict_proba(input_scaled)[0]
            else:
                proba = None

            st.markdown(f"""
            <div class="prediction-result">
                <p style="font-size:3em;">{emoji_map.get(species_name, '🌸')}</p>
                <p class="prediction-species">{species_name.title()}</p>
                <p style="color:#666;">Species Index: {prediction}</p>
            </div>
            """, unsafe_allow_html=True)

            if proba is not None:
                st.markdown("#### Prediction Confidence")
                conf_df = pd.DataFrame({
                    "Species": TARGET_NAMES,
                    "Confidence": proba,
                })
                st.dataframe(
                    conf_df.style.bar(subset=["Confidence"], color="#4caf50"),
                    use_container_width=True, hide_index=True
                )

                st.markdown("#### Confidence Chart")
                st.bar_chart(conf_df.set_index("Species"))


if __name__ == "__main__":
    main()
