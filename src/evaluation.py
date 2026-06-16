import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score

TARGET_NAMES = ["setosa", "versicolor", "virginica"]
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def evaluate_all(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({
            "Model": name,
            "Test Accuracy": acc,
        })
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return df_results


def evaluate_with_cv(models, X_train, y_train, X_test, y_test, cv=5):
    results = []
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        results.append({
            "Model": name,
            "CV Mean Accuracy": cv_mean,
            "CV Std": cv_std,
            "Test Accuracy": test_acc,
        })
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    return df_results


def get_classification_reports(models, X_test, y_test):
    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        reports[name] = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
    return reports


def plot_confusion_matrices(models, X_test, y_test):
    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                    linewidths=0.5, ax=axes[idx], cbar=False)
        axes[idx].set_title(name, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_roc_curves(models, X_test, y_test):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]

    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    line_styles = ["-", "--", "-."]
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        else:
            y_score = model.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.column_stack([1 - y_score, y_score])

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax.plot(fpr[i], tpr[i], linestyle=line_styles[i % len(line_styles)],
                    linewidth=2, label=f"{TARGET_NAMES[i]} (AUC={roc_auc[i]:.3f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name} \u2014 ROC Curves", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(linestyle="--", alpha=0.3)

    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("One-vs-Rest ROC Curves", fontweight="bold", fontsize=14)
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    df = results_df.copy()
    acc_col = [c for c in df.columns if "Accuracy" in c and "Test" in c][0]
    df = df.sort_values(acc_col)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df["Model"], df[acc_col], color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, df[acc_col]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", va="center", fontweight="bold", fontsize=10)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("Model Test Accuracy Comparison", fontweight="bold", fontsize=13)
    sns.despine()
    plt.tight_layout()
    return fig


def plot_cv_comparison(results_df):
    fig, ax = plt.subplots(figsize=(9, 4))
    df = results_df.copy()
    df = df.sort_values("CV Mean Accuracy")
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df["Model"], df["CV Mean Accuracy"], color=colors,
                   xerr=df["CV Std"], capsize=5, edgecolor="black", linewidth=0.8)
    for bar, val, std_val in zip(bars, df["CV Mean Accuracy"], df["CV Std"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}\u00b1{std_val:.2%}", va="center", fontweight="bold", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Cross-Validation Accuracy (5-Fold)")
    ax.set_title("5-Fold Cross-Validation Accuracy Comparison", fontweight="bold", fontsize=13)
    sns.despine()
    plt.tight_layout()
    return fig
