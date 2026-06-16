import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

TARGET_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10


def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=FEATURE_NAMES)
    df["species"] = iris.target
    df["species_name"] = df["species"].map(TARGET_NAMES)
    return df


def display_shape(df):
    return df.shape


def display_info(df):
    return df.info(verbose=True, show_counts=True)


def display_describe(df):
    return df.describe()


def display_null_counts(df):
    return df.isnull().sum()


def display_head(df, n=5):
    return df.head(n)


def display_class_distribution(df):
    return df["species_name"].value_counts()


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["species_name"].value_counts()
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Class Distribution — Iris Species", fontweight="bold", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_xlabel("Species")
    ax.set_ylim(0, max(counts.values) * 1.15)
    sns.despine()
    plt.tight_layout()
    return fig


def plot_pairplot(df):
    g = sns.pairplot(
        df, hue="species_name", vars=FEATURE_NAMES,
        palette={"setosa": "#3498db", "versicolor": "#2ecc71", "virginica": "#e74c3c"},
        diag_kind="hist", plot_kws={"alpha": 0.7, "s": 40, "edgecolor": "k", "linewidth": 0.3},
        height=2.2
    )
    g.fig.suptitle("Pairplot — Iris Features by Species", y=1.02, fontweight="bold", fontsize=14)
    return g.fig


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[FEATURE_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
                vmin=-1, vmax=1, linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontweight="bold", fontsize=13)
    plt.tight_layout()
    return fig


def plot_pca_2d(df):
    X = df[FEATURE_NAMES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["species"] = df["species_name"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"setosa": "#3498db", "versicolor": "#2ecc71", "virginica": "#e74c3c"}
    for species in pca_df["species"].unique():
        subset = pca_df[pca_df["species"] == species]
        ax.scatter(subset["PC1"], subset["PC2"], c=colors[species],
                   label=species, edgecolor="k", s=50, alpha=0.8)
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA — 2D Projection of Iris Dataset", fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    return fig


def plot_pca_3d(df):
    X = df[FEATURE_NAMES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2", "PC3"])
    pca_df["species"] = df["species_name"].values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"setosa": "#3498db", "versicolor": "#2ecc71", "virginica": "#e74c3c"}
    markers = {"setosa": "o", "versicolor": "^", "virginica": "s"}

    for species in pca_df["species"].unique():
        subset = pca_df[pca_df["species"] == species]
        ax.scatter(
            subset["PC1"], subset["PC2"], subset["PC3"],
            c=colors[species], marker=markers[species],
            label=species, edgecolor="k", s=60, alpha=0.85, depthshade=True
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=10)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=10)
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})", fontsize=10)
    ax.set_title("PCA — 3D Projection of Iris Dataset", fontweight="bold", fontsize=14, pad=20)
    ax.legend(loc="upper right", fontsize=9)

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    return fig


def plot_explained_variance():
    iris = load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)
    n_components = range(1, len(explained_var) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(n_components, explained_var, color="steelblue",
                  edgecolor="black", linewidth=0.8, label="Individual explained variance",
                  width=0.55)

    ax.step(n_components, cum_var, where="mid", color="red", linewidth=2,
            linestyle="--", label="Cumulative explained variance", zorder=5)

    ax.scatter(n_components, cum_var, color="red", s=60, zorder=10)

    for i, (ev, cv) in enumerate(zip(explained_var, cum_var)):
        ax.text(n_components[i], ev + 0.01, f"{ev:.1%}", ha="center",
                va="bottom", fontsize=9, fontweight="bold")
        ax.text(n_components[i], cv + 0.02, f"{cv:.1%}", ha="center",
                va="bottom", fontsize=8, color="red")

    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance Ratio", fontsize=11)
    ax.set_title("PCA Explained Variance — Iris Dataset", fontweight="bold", fontsize=13)
    ax.set_xticks(n_components)
    ax.set_xticklabels([f"PC{i}" for i in n_components])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="center right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    sns.despine()
    plt.tight_layout()
    return fig
