from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data_demo" / "demo_pairs.csv"
RESULTS_PATH = REPO_ROOT / "experiments" / "results_table.csv"
FIGURES_DIR = REPO_ROOT / "figures"


def feature_columns(df: pd.DataFrame, prefixes: list[str]) -> list[str]:
    for prefix in prefixes:
        cols = sorted([c for c in df.columns if c.startswith(prefix)])
        if cols:
            return cols

    raise RuntimeError(f"No feature columns found for prefixes: {prefixes}")


def detect_cluster_column(df: pd.DataFrame) -> str:
    for col in ["cluster_id", "semantic_cluster", "group_id", "label"]:
        if col in df.columns:
            return col
    raise RuntimeError("No cluster column found. Expected cluster_id, semantic_cluster, group_id, or label.")


def pretty_loss(row: pd.Series) -> str:
    if row["loss"] == "standard":
        return "Standard"
    alpha = row.get("alpha")
    if pd.isna(alpha) or alpha == "":
        return "FN-aware"
    return f"FN-aware α={alpha}"


def plot_input_feature_space(max_points: int = 2000, seed: int = 42) -> None:
    df = pd.read_csv(DATA_PATH)

    a_cols = feature_columns(df, ["a_feat_", "modality_a_", "a_"])
    b_cols = feature_columns(df, ["b_feat_", "modality_b_", "b_"])
    cluster_col = detect_cluster_column(df)

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed).sort_index().reset_index(drop=True)

    a_x = df[a_cols].astype(np.float32).to_numpy()
    b_x = df[b_cols].astype(np.float32).to_numpy()
    clusters = df[cluster_col].astype(int).to_numpy()

    combined = np.vstack([a_x, b_x])
    coords = PCA(n_components=2, random_state=42).fit_transform(combined)

    a_z = coords[: len(a_x)]
    b_z = coords[len(a_x) :]

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    ax.scatter(
        a_z[:, 0],
        a_z[:, 1],
        c=clusters,
        cmap="tab20",
        s=9,
        alpha=0.70,
        marker="o",
        linewidths=0,
        label="Modality A",
    )

    ax.scatter(
        b_z[:, 0],
        b_z[:, 1],
        c=clusters,
        cmap="tab20",
        s=10,
        alpha=0.70,
        marker="x",
        linewidths=0.5,
        label="Modality B",
    )

    ax.set_title("Synthetic paired feature space")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fn_aware_input_feature_space.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_recall_comparison(n_samples: int = 20000) -> None:
    df = pd.read_csv(RESULTS_PATH)
    df = df[df["n_samples"] == n_samples].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for n_samples={n_samples}")

    df["method_label"] = df.apply(pretty_loss, axis=1)

    mode_order = ["clean", "clustered", "noisy"]
    method_order = ["Standard", "FN-aware α=0.5", "FN-aware α=0.25"]

    df["mode"] = pd.Categorical(df["mode"], categories=mode_order, ordered=True)
    df["method_label"] = pd.Categorical(df["method_label"], categories=method_order, ordered=True)
    df = df.sort_values(["mode", "method_label"])

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    x = np.arange(len(mode_order))
    width = 0.24

    for i, method in enumerate(method_order):
        sub = df[df["method_label"] == method]
        values = []
        for mode in mode_order:
            row = sub[sub["mode"] == mode]
            values.append(float(row["val_recall@1"].iloc[0]) if len(row) else np.nan)

        ax.bar(x + (i - 1) * width, values, width=width, label=method)

    ax.set_title(f"Recall@1 comparison ({n_samples} samples)")
    ax.set_ylabel("Validation Recall@1")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "Clustered", "Noisy"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fn_aware_recall_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_similarity_diagnostic(n_samples: int = 20000, mode: str = "clustered") -> None:
    df = pd.read_csv(RESULTS_PATH)
    df = df[(df["n_samples"] == n_samples) & (df["mode"] == mode)].copy()

    if df.empty:
        raise RuntimeError(f"No rows found for mode={mode}, n_samples={n_samples}")

    df["method_label"] = df.apply(pretty_loss, axis=1)

    method_order = ["Standard", "FN-aware α=0.5", "FN-aware α=0.25"]
    df["method_label"] = pd.Categorical(df["method_label"], categories=method_order, ordered=True)
    df = df.sort_values("method_label")

    metrics = [
        ("val_pos_sim_mean", "Positive pairs"),
        ("val_eval_same_cluster_neg_sim", "Same-cluster negatives"),
        ("val_eval_diff_cluster_neg_sim", "Different-cluster negatives"),
    ]

    x = np.arange(len(method_order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    for i, (col, label) in enumerate(metrics):
        values = []
        for method in method_order:
            row = df[df["method_label"] == method]
            values.append(float(row[col].iloc[0]) if len(row) else np.nan)

        ax.bar(x + (i - 1) * width, values, width=width, label=label)

    ax.set_title(f"Similarity diagnostic in {mode} mode ({n_samples} samples)")
    ax.set_ylabel("Cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fn_aware_similarity_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_input_feature_space()
    plot_recall_comparison(n_samples=20000)
    plot_similarity_diagnostic(n_samples=20000, mode="clustered")

    print("Saved figures:")
    print(FIGURES_DIR / "fn_aware_input_feature_space.png")
    print(FIGURES_DIR / "fn_aware_recall_comparison.png")
    print(FIGURES_DIR / "fn_aware_similarity_diagnostic.png")


if __name__ == "__main__":
    main()