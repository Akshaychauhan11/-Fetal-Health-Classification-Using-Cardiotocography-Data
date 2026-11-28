"""
Global (Marginal) Split Conformal Prediction for Fetal Health Classification.

Provides standard split conformal prediction using MAPIE:
- LAC (Least Ambiguous Classifier): conformity score based on maximum class probability
- APS (Adaptive Prediction Sets): cumulative probability-based conformity scoring (Romano et al.)

Evaluates:
- Marginal (overall) coverage vs. theoretical guarantee (1 - alpha)
- Per-class coverage (exposing minority class under-coverage in imbalanced data)
- Prediction set size efficiency and set composition (empty, singleton, multi-class)
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import LACConformityScore, APSConformityScore

# Default class names for CTG dataset
CLASS_NAMES_DEFAULT = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


def fit_global_conformal_classifier(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    method: str = "lac",
    confidence_level: Union[float, List[float]] = 0.90,
    random_state: int = 42
) -> SplitConformalClassifier:
    """
    Initialize and conformalize a SplitConformalClassifier using pre-fit base model.

    Parameters
    ----------
    estimator : fitted classifier (e.g. RandomForestClassifier, XGBClassifier)
    X_cal : array-like of shape (n_cal, n_features), calibration feature set
    y_cal : array-like of shape (n_cal,), calibration true labels (1.0, 2.0, 3.0)
    method : str, 'lac' or 'aps'
    confidence_level : float or list of floats, e.g. 0.90 or [0.80, 0.90, 0.95]
    random_state : int, seed for reproducibility

    Returns
    -------
    SplitConformalClassifier ready for prediction
    """
    method_lower = method.lower()
    if method_lower == "lac":
        score = LACConformityScore()
    elif method_lower == "aps":
        score = APSConformityScore()
    else:
        raise ValueError(f"Unknown method '{method}'. Supported methods: 'lac', 'aps'.")

    # In MAPIE 1.5.0+, prefit=True requires calling .conformalize() rather than .fit()
    cp = SplitConformalClassifier(
        estimator=estimator,
        conformity_score=score,
        confidence_level=confidence_level,
        prefit=True,
        random_state=random_state
    )
    cp.conformalize(X_cal, y_cal)
    return cp


def predict_prediction_sets(
    cp_classifier: SplitConformalClassifier,
    X: Union[pd.DataFrame, np.ndarray],
    conformity_score_params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict point labels and boolean prediction sets for input samples.

    Parameters
    ----------
    cp_classifier : conformalized SplitConformalClassifier
    X : array-like of shape (n_samples, n_features)
    conformity_score_params : optional parameters for conformity score (e.g. include_last_label)

    Returns
    -------
    y_pred : ndarray of shape (n_samples,) point predictions
    y_ps : ndarray of shape (n_samples, n_classes, n_confidence_levels) or (n_samples, n_classes)
           boolean array where True indicates class inclusion in the prediction set
    """
    y_pred, y_ps = cp_classifier.predict_set(X, conformity_score_params=conformity_score_params)
    return y_pred, y_ps


def evaluate_conformal_coverage(
    y_true: Union[pd.Series, np.ndarray],
    y_ps: np.ndarray,
    classes: np.ndarray,
    class_names: Optional[Dict[float, str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute overall marginal coverage, per-class coverage, average set size,
    and prediction set size distribution (empty, singleton, multi-class).

    Parameters
    ----------
    y_true : 1D array of true labels
    y_ps : 2D boolean array of shape (n_samples, n_classes)
    classes : array of class values matching columns of y_ps (e.g. [1.0, 2.0, 3.0])
    class_names : optional mapping of class float/int to string name

    Returns
    -------
    dict with metrics:
        - marginal_coverage: float
        - per_class_coverage: dict of {class_name: coverage}
        - avg_set_size: float
        - per_class_avg_set_size: dict of {class_name: avg_size}
        - empty_set_pct: percentage of samples with 0 labels
        - singleton_set_pct: percentage of samples with 1 label
        - multi_class_set_pct: percentage of samples with >= 2 labels
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    y_arr = np.asarray(y_true)
    if y_ps.ndim == 3:
        if y_ps.shape[2] == 1:
            y_ps = y_ps[:, :, 0]
        else:
            raise ValueError(
                f"y_ps contains {y_ps.shape[2]} confidence levels (shape: {y_ps.shape}). "
                "evaluate_conformal_coverage evaluates a single confidence level at a time. "
                "Pass a 2D slice y_ps[:, :, i] or iterate across confidence levels."
            )

    n_samples = len(y_arr)
    set_sizes = np.sum(y_ps, axis=1)

    # Find column index for each sample's true class
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[y] for y in y_arr])

    # Check whether true class is inside the prediction set
    correct_in_set = np.array([y_ps[i, y_indices[i]] for i in range(n_samples)], dtype=bool)

    marginal_cov = float(np.mean(correct_in_set))
    avg_size = float(np.mean(set_sizes))

    # Per-class breakdown
    per_class_cov = {}
    per_class_size = {}
    for c in classes:
        mask = (y_arr == c)
        c_name = class_names.get(c, str(c))
        if np.sum(mask) > 0:
            per_class_cov[c_name] = float(np.mean(correct_in_set[mask]))
            per_class_size[c_name] = float(np.mean(set_sizes[mask]))
        else:
            per_class_cov[c_name] = float("nan")
            per_class_size[c_name] = float("nan")

    # Set composition
    empty_pct = float(np.mean(set_sizes == 0) * 100.0)
    singleton_pct = float(np.mean(set_sizes == 1) * 100.0)
    multi_pct = float(np.mean(set_sizes >= 2) * 100.0)

    return {
        "marginal_coverage": round(marginal_cov, 4),
        "per_class_coverage": {k: round(v, 4) for k, v in per_class_cov.items()},
        "avg_set_size": round(avg_size, 4),
        "per_class_avg_set_size": {k: round(v, 4) for k, v in per_class_size.items()},
        "empty_set_pct": round(empty_pct, 2),
        "singleton_set_pct": round(singleton_pct, 2),
        "multi_class_set_pct": round(multi_pct, 2),
    }


def sweep_alpha_conformal(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    method: str = "lac",
    alphas: Optional[List[float]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sweep across a range of significance levels alpha (or confidence levels 1 - alpha)
    and compute marginal coverage, per-class coverage, average set size, and composition.

    Parameters
    ----------
    estimator : fitted base classifier
    X_cal, y_cal : calibration split
    X_test, y_test : test split
    method : 'lac' or 'aps'
    alphas : list of significance levels (default: [0.01, 0.05, 0.10, 0.15, 0.20, 0.25])
    random_state : int

    Returns
    -------
    pd.DataFrame with evaluation metrics for each alpha
    """
    if alphas is None:
        alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    confidence_levels = [round(1.0 - a, 4) for a in alphas]
    classes = np.asarray(getattr(estimator, "classes_", sorted(list(set(y_cal)))))

    rows = []
    # Train conformal classifier with all target confidence levels
    cp = fit_global_conformal_classifier(
        estimator=estimator,
        X_cal=X_cal,
        y_cal=y_cal,
        method=method,
        confidence_level=confidence_levels,
        random_state=random_state
    )

    y_pred, y_ps_all = predict_prediction_sets(cp, X_test)

    for i, alpha in enumerate(alphas):
        target_conf = confidence_levels[i]
        # y_ps_all shape is (n_samples, n_classes, n_confidence_levels)
        ps_alpha = y_ps_all[:, :, i]

        metrics = evaluate_conformal_coverage(
            y_true=y_test,
            y_ps=ps_alpha,
            classes=classes
        )

        row = {
            "method": method.upper(),
            "alpha": alpha,
            "target_confidence": target_conf,
            "empirical_coverage": metrics["marginal_coverage"],
            "coverage_gap": round(metrics["marginal_coverage"] - target_conf, 4),
            "avg_set_size": metrics["avg_set_size"],
            "cov_Normal": metrics["per_class_coverage"].get("Normal", float("nan")),
            "cov_Suspect": metrics["per_class_coverage"].get("Suspect", float("nan")),
            "cov_Pathological": metrics["per_class_coverage"].get("Pathological", float("nan")),
            "empty_pct": metrics["empty_set_pct"],
            "singleton_pct": metrics["singleton_set_pct"],
            "multi_pct": metrics["multi_class_set_pct"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_global_conformal_comparison(
    df_lac: pd.DataFrame,
    df_aps: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Create a 4-panel publication-grade figure comparing LAC vs. APS:
    1. Target vs. Empirical Marginal Coverage (Calibration Line)
    2. Average Prediction Set Size vs. Confidence Level (Efficiency)
    3. Per-Class Coverage at Alpha=0.10 (Highlighting Minority Class Under-Coverage)
    4. Prediction Set Composition Breakdown (Empty vs. Singleton vs. Multi-class)
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Target vs Empirical Marginal Coverage
    ax1 = axes[0, 0]
    ax1.plot(df_lac["target_confidence"], df_lac["empirical_coverage"], marker="o", label="LAC Empirical", color="#1f77b4", linewidth=2)
    ax1.plot(df_aps["target_confidence"], df_aps["empirical_coverage"], marker="s", label="APS Empirical", color="#ff7f0e", linewidth=2)
    min_c = min(df_lac["target_confidence"].min(), df_aps["target_confidence"].min())
    max_c = max(df_lac["target_confidence"].max(), df_aps["target_confidence"].max())
    ax1.plot([min_c, max_c], [min_c, max_c], "k--", label="Ideal Guarantee (y = x)", alpha=0.7)
    ax1.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax1.set_ylabel("Empirical Marginal Coverage", fontweight="bold")
    ax1.set_title("Panel A: Coverage Validity (Global LAC vs. APS)", fontweight="bold", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.set_ylim([0.70, 1.02])

    # Panel 2: Average Set Size vs Confidence Level (Efficiency)
    ax2 = axes[0, 1]
    ax2.plot(df_lac["target_confidence"], df_lac["avg_set_size"], marker="o", label="LAC Avg Set Size", color="#1f77b4", linewidth=2)
    ax2.plot(df_aps["target_confidence"], df_aps["avg_set_size"], marker="s", label="APS Avg Set Size", color="#ff7f0e", linewidth=2)
    ax2.axhline(1.0, color="gray", linestyle=":", label="Single Label (|S|=1)")
    ax2.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax2.set_ylabel("Average Prediction Set Size", fontweight="bold")
    ax2.set_title("Panel B: Efficiency (Average Set Size vs. Target Coverage)", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left")

    # Panel 3: Per-Class Coverage at alpha=0.10 (Target = 0.90)
    ax3 = axes[1, 0]
    # Filter alpha = 0.10
    lac_10 = df_lac[np.isclose(df_lac["alpha"], 0.10)].iloc[0]
    aps_10 = df_aps[np.isclose(df_aps["alpha"], 0.10)].iloc[0]

    classes_list = ["Normal", "Suspect", "Pathological"]
    lac_covs = [lac_10["cov_Normal"], lac_10["cov_Suspect"], lac_10["cov_Pathological"]]
    aps_covs = [aps_10["cov_Normal"], aps_10["cov_Suspect"], aps_10["cov_Pathological"]]

    x = np.arange(len(classes_list))
    width = 0.35
    ax3.bar(x - width/2, lac_covs, width, label="Global LAC", color="#1f77b4", alpha=0.85)
    ax3.bar(x + width/2, aps_covs, width, label="Global APS", color="#ff7f0e", alpha=0.85)
    ax3.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="Target 90% Guarantee")
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes_list, fontweight="bold")
    ax3.set_ylabel("Empirical Class Coverage", fontweight="bold")
    ax3.set_title("Panel C: Per-Class Coverage at α = 0.10 (The Minority Gap)", fontweight="bold", fontsize=11)
    ax3.set_ylim([0.0, 1.08])
    ax3.legend(loc="lower left")

    # Annotate values
    for i, v in enumerate(lac_covs):
        ax3.text(i - width/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(aps_covs):
        ax3.text(i + width/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")

    # Panel 4: Set Composition Breakdown for LAC across Alphas (Categorical bars to avoid overlapping)
    ax4 = axes[1, 1]
    n_bars = len(df_lac)
    x_indices = np.arange(n_bars)
    bar_w = 0.55
    ax4.bar(x_indices, df_lac["empty_pct"], width=bar_w, label="Empty Set (|S|=0)", color="#d62728", alpha=0.85)
    ax4.bar(x_indices, df_lac["singleton_pct"], width=bar_w, bottom=df_lac["empty_pct"], label="Singleton (|S|=1)", color="#2ca02c", alpha=0.85)
    ax4.bar(x_indices, df_lac["multi_pct"], width=bar_w, bottom=df_lac["empty_pct"] + df_lac["singleton_pct"], label="Ambiguous (|S|>=2)", color="#9467bd", alpha=0.85)

    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"{c:.2f}" for c in df_lac["target_confidence"]], fontweight="bold")
    ax4.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax4.set_ylabel("Set Composition (%)", fontweight="bold")
    ax4.set_title("Panel D: Global LAC Prediction Set Composition", fontweight="bold", fontsize=11)
    ax4.set_ylim([0, 105])
    ax4.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.conformal.global_cp] Saved comparison figure to: {save_path}")

    return fig
