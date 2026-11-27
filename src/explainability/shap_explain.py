"""
SHAP-based Explainability Module for Fetal Health Classification.

Provides:
1. TreeExplainer initialization for tree-based models (Random Forest, XGBoost).
2. Local explanation extraction: returns top-k contributing features for individual patient predictions.
3. Global feature importance generation for accepted predictions.
4. Publication-grade plotting for global beeswarm / bar summaries and local waterfall explanations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

CLASS_NAMES_DEFAULT = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


def get_tree_explainer(model) -> shap.TreeExplainer:
    """
    Initialize and return a shap.TreeExplainer for the fitted tree model.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: Union[pd.DataFrame, np.ndarray]
) -> shap.Explanation:
    """
    Compute SHAP values for the given feature matrix X.
    Returns shap.Explanation of shape (n_samples, n_features, n_classes).
    """
    return explainer(X)


def get_top_features_for_sample(
    shap_values: shap.Explanation,
    feature_names: List[str],
    sample_idx: int = 0,
    class_idx: int = 0,
    top_n: int = 5
) -> List[Dict[str, Union[str, float]]]:
    """
    Extract top N features contributing to the prediction for a specific sample.

    Parameters
    ----------
    shap_values : shap.Explanation object
    feature_names : list of feature names matching X columns
    sample_idx : index of the sample in shap_values
    class_idx : target class index (0 for Normal, 1 for Suspect, 2 for Pathological)
    top_n : number of top features to return

    Returns
    -------
    list of dicts containing:
        - 'feature': feature name
        - 'shap_value': SHAP impact (positive pushes toward class, negative pushes away)
        - 'feature_value': raw feature measurement
    """
    sample_shap = shap_values[sample_idx, :, class_idx]
    values = sample_shap.values
    data = sample_shap.data if hasattr(sample_shap, "data") else None

    # Sort by absolute SHAP impact
    top_indices = np.argsort(np.abs(values))[::-1][:top_n]

    results = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        shap_val = float(values[idx])
        feat_val = float(data[idx]) if data is not None else float("nan")
        results.append({
            "feature": feat_name,
            "shap_value": round(shap_val, 4),
            "feature_value": round(feat_val, 4)
        })

    return results


def get_top_feature_names_only(
    shap_values: shap.Explanation,
    feature_names: List[str],
    sample_idx: int = 0,
    class_idx: int = 0,
    top_n: int = 3
) -> List[str]:
    """
    Convenience function returning just the list of top N feature name strings
    for the FastAPI JSON response (e.g. ['abnormal_short_term_variability', 'accelerations']).
    """
    top_dicts = get_top_features_for_sample(
        shap_values=shap_values,
        feature_names=feature_names,
        sample_idx=sample_idx,
        class_idx=class_idx,
        top_n=top_n
    )
    return [d["feature"] for d in top_dicts]


def plot_global_feature_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
    class_names: Optional[Dict[float, str]] = None,
    top_n: int = 12,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate publication-grade summary of global feature importance:
    Stacked horizontal bar chart showing mean absolute SHAP values per class.
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 7))

    # Mean absolute SHAP values per class: shape (n_features, n_classes)
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0) # shape (21, 3)
    total_importance = np.sum(mean_abs_shap, axis=1)

    # Sort by total importance
    top_indices = np.argsort(total_importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices][::-1]
    top_shap = mean_abs_shap[top_indices][::-1]

    colors = ["#2ca02c", "#ff7f0e", "#d62728"] # Normal (green), Suspect (orange), Patho (red)
    labels = ["Normal (Class 1)", "Suspect (Class 2)", "Pathological (Class 3)"]

    y_pos = np.arange(len(top_features))
    left = np.zeros(len(top_features))

    for k in range(3):
        ax.barh(y_pos, top_shap[:, k], left=left, color=colors[k], label=labels[k], alpha=0.85, height=0.65)
        left += top_shap[:, k]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontweight="bold")
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Confidence)", fontweight="bold")
    ax.set_title("Global Feature Importance for Fetal Health Classification (SHAP Summary)", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.explainability.shap_explain] Saved global feature importance to: {save_path}")

    return fig


def plot_local_patient_explanation(
    shap_values: shap.Explanation,
    feature_names: List[str],
    sample_idx: int,
    class_idx: int,
    class_name: str,
    top_n: int = 8,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate horizontal bar explanation for a single patient:
    Shows positive pushes (toward diagnosis) and negative pushes (away from diagnosis).
    """
    sample_shap = shap_values[sample_idx, :, class_idx]
    values = sample_shap.values
    data = sample_shap.data if hasattr(sample_shap, "data") else None

    top_indices = np.argsort(np.abs(values))[::-1][:top_n]
    feats = [feature_names[i] for i in top_indices][::-1]
    shaps = [values[i] for i in top_indices][::-1]
    raw_vals = [data[i] for i in top_indices][::-1] if data is not None else [None] * len(feats)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    colors = ["#d62728" if s < 0 else "#2ca02c" for s in shaps]
    y_pos = np.arange(len(feats))

    bars = ax.barh(y_pos, shaps, color=colors, alpha=0.85, height=0.6)
    ax.axvline(0.0, color="black", linestyle="--", alpha=0.7)

    y_labels = []
    for i, f in enumerate(feats):
        val_str = f" = {raw_vals[i]:.2f}" if raw_vals[i] is not None else ""
        y_labels.append(f"{f}{val_str}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold")
    ax.set_xlabel(f"SHAP Value (Contribution toward '{class_name}')", fontweight="bold")
    ax.set_title(f"Clinical Case Explanation: Patient #{sample_idx} -> Diagnosed as '{class_name}'", fontweight="bold", fontsize=11)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        offset = 0.005 if w >= 0 else -0.005
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.3f}",
                va="center", ha=ha, fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.explainability.shap_explain] Saved local patient explanation to: {save_path}")

    return fig
