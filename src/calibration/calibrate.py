"""
Calibration evaluation metrics and visualization for Fetal Health Classification.

Evaluates how trustworthy raw predicted probabilities are:
1. Multi-class Brier Score (overall and per-class one-vs-rest)
2. Expected Calibration Error (ECE) for top-class confidence
3. Class-wise Expected Calibration Error (ECE per class)
4. Reliability Diagram plotting (calibration curves vs 45-degree perfect line)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from sklearn.calibration import calibration_curve


def _to_zero_indexed_and_onehot(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    n_classes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize class labels to 0-indexed integers [0, 1, ..., K-1]
    and return both the 1D integer array and 2D one-hot encoded matrix.
    """
    y_arr = np.asarray(y_true, dtype=int)
    if y_arr.min() == 1:
        y_arr = y_arr - 1

    # One-hot encode
    n_samples = len(y_arr)
    one_hot = np.zeros((n_samples, n_classes), dtype=float)
    one_hot[np.arange(n_samples), y_arr] = 1.0

    return y_arr, one_hot


def compute_brier_scores(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_probs: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multi-class Brier score and per-class one-vs-rest Brier scores.

    Formula:
        Brier_multiclass = (1/N) * sum_i sum_k (p_ik - y_ik)^2
        Brier_class_k   = (1/N) * sum_i (p_ik - y_ik)^2

    Parameters
    ----------
    y_true : 1D array-like of true class labels (can be 1-indexed or 0-indexed).
    y_probs : 2D array of shape (N, K) of predicted class probabilities.
    class_names : Optional list of class names (default: ['Normal', 'Suspect', 'Pathological']).

    Returns
    -------
    Dict[str, float] containing:
        - 'brier_multiclass': overall multi-class Brier score
        - 'brier_<class_name>': one-vs-rest Brier score for each class
    """
    if class_names is None:
        class_names = ["Normal", "Suspect", "Pathological"]

    n_classes = len(class_names)
    y_idx, y_onehot = _to_zero_indexed_and_onehot(y_true, n_classes=n_classes)

    # Multi-class Brier: mean sum of squared probability errors across all classes
    # Range: 0.0 (perfect) to 2.0 (complete misclassification with 100% confidence)
    squared_diffs = (y_probs - y_onehot) ** 2
    multiclass_brier = float(np.mean(np.sum(squared_diffs, axis=1)))

    results = {"brier_multiclass": round(multiclass_brier, 4)}

    # Per-class one-vs-rest Brier scores
    for k, name in enumerate(class_names):
        class_brier = float(np.mean(squared_diffs[:, k]))
        results[f"brier_{name}"] = round(class_brier, 4)

    return results


def compute_ece(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_probs: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    based on the top-label confidence: max_k P(Y = k | X).

    Partitions [0, 1] into M equal-width bins:
        ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
        MCE = max_m |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : 1D array-like of true labels.
    y_probs : 2D array of shape (N, K) of predicted probabilities.
    n_bins : Number of confidence bins (default: 10).

    Returns
    -------
    Dict[str, float] containing:
        - 'ece': Expected Calibration Error
        - 'mce': Maximum Calibration Error
    """
    y_idx, _ = _to_zero_indexed_and_onehot(y_true, n_classes=y_probs.shape[1])

    # Top-1 predicted class and associated confidence
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_idx).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(y_idx)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Samples falling into current bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            calibration_gap = abs(bin_acc - bin_conf)

            ece += (bin_size / n_samples) * calibration_gap
            mce = max(mce, calibration_gap)

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4)
    }


def compute_classwise_ece(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_probs: np.ndarray,
    n_bins: int = 10,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute marginal (one-vs-rest) Expected Calibration Error for each class.
    Essential for imbalanced medical data to detect if the minority class (Pathological)
    is specifically overconfident or underconfident.

    Parameters
    ----------
    y_true : 1D array-like of true labels.
    y_probs : 2D array of predicted probabilities.
    n_bins : Number of confidence bins (default: 10).
    class_names : List of class names.

    Returns
    -------
    Dict[str, float] containing 'ece_<class_name>' for each class.
    """
    if class_names is None:
        class_names = ["Normal", "Suspect", "Pathological"]

    n_classes = len(class_names)
    y_idx, y_onehot = _to_zero_indexed_and_onehot(y_true, n_classes=n_classes)
    n_samples = len(y_idx)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    class_ece = {}

    for k, name in enumerate(class_names):
        class_probs = y_probs[:, k]
        class_actual = y_onehot[:, k]

        ece_k = 0.0
        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]

            if i == n_bins - 1:
                in_bin = (class_probs >= bin_lower) & (class_probs <= bin_upper)
            else:
                in_bin = (class_probs >= bin_lower) & (class_probs < bin_upper)

            bin_size = np.sum(in_bin)
            if bin_size > 0:
                bin_acc = np.mean(class_actual[in_bin])
                bin_conf = np.mean(class_probs[in_bin])
                ece_k += (bin_size / n_samples) * abs(bin_acc - bin_conf)

        class_ece[f"ece_{name}"] = round(float(ece_k), 4)

    return class_ece


def evaluate_calibration(
    y_true: Union[np.ndarray, pd.Series, List[int]],
    y_probs: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    class_names: Optional[List[str]] = None
) -> Dict[str, Union[str, float]]:
    """
    Run full calibration diagnostic suite:
    - Multi-class Brier Score
    - Per-class Brier Scores
    - Top-label ECE & MCE
    - Per-class ECE
    """
    if class_names is None:
        class_names = ["Normal", "Suspect", "Pathological"]

    brier = compute_brier_scores(y_true, y_probs, class_names=class_names)
    ece_metrics = compute_ece(y_true, y_probs, n_bins=n_bins)
    class_ece = compute_classwise_ece(y_true, y_probs, n_bins=n_bins, class_names=class_names)

    summary = {"model": model_name}
    summary.update(brier)
    summary.update(ece_metrics)
    summary.update(class_ece)

    return summary


def plot_reliability_diagrams(
    models_dict: Dict[str, np.ndarray],
    y_true: Union[np.ndarray, pd.Series, List[int]],
    class_names: Optional[List[str]] = None,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 3-panel reliability diagrams (one panel per class: Normal, Suspect, Pathological)
    comparing calibration curves of all models against the perfect calibration diagonal.

    Parameters
    ----------
    models_dict : Dict mapping model_name (str) -> predicted_probabilities (N, 3).
    y_true : Ground truth class labels for calibration set.
    class_names : Class display names.
    n_bins : Number of bins for calibration curves.
    save_path : Optional filepath to save the resulting figure.

    Returns
    -------
    matplotlib.figure.Figure object.
    """
    if class_names is None:
        class_names = ["Normal (1)", "Suspect (2)", "Pathological (3)"]

    n_classes = len(class_names)
    y_idx, y_onehot = _to_zero_indexed_and_onehot(y_true, n_classes=n_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5.5), sharey=True)

    colors = {
        "Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728"
    }

    for k, class_name in enumerate(class_names):
        ax = axes[k]
        # Diagonal reference line: perfectly calibrated
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

        y_actual = y_onehot[:, k]

        for model_name, y_probs in models_dict.items():
            probs_k = y_probs[:, k]

            # Compute calibration curve points using sklearn
            prob_true, prob_pred = calibration_curve(
                y_actual, probs_k, n_bins=n_bins, strategy="uniform"
            )

            color = colors.get(model_name, None)
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                linewidth=2.2,
                markersize=6,
                label=model_name,
                color=color
            )

        ax.set_title(f"Class: {class_name}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        if k == 0:
            ax.set_ylabel("Empirical Fraction of Positives", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    fig.suptitle(
        "Reliability Diagrams (Calibration Curves) — Calibration Set (N=425)",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.calibration.calibrate] Figure saved successfully to: {save_path}")

    return fig
