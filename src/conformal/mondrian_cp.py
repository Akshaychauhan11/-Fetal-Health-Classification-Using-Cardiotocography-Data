"""
Mondrian (Class-Conditional) Conformal Prediction and Calibration-Size Sensitivity.

Provides:
1. MondrianConformalClassifier: Class-conditional split conformal prediction
   guaranteeing 1 - alpha coverage independently for EVERY class, including
   the high-risk minority Pathological class.
2. Sensitivity Analysis (RQ3): Monte-Carlo subsampling to measure how calibration
   set size impacts coverage stability and prediction set variance (mean +/- std).
3. Publication-grade plotting for Mondrian comparisons and sensitivity curves.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# Default class names for CTG dataset
CLASS_NAMES_DEFAULT = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


class MondrianConformalClassifier:
    """
    Class-Conditional (Mondrian) Split Conformal Classifier.

    Rather than computing a single global non-conformity threshold across all classes,
    Mondrian conformal prediction partitions the calibration set by class and calculates
    an independent non-conformity quantile q_k for each class k in {1, ..., K}.

    Guarantees:
        P(Y in Gamma(X) | Y = k) >= 1 - alpha  for every class k in Y.
    """

    def __init__(
        self,
        estimator,
        alpha: float = 0.10,
        random_state: int = 42
    ):
        self.estimator = estimator
        self.alpha = float(alpha)
        self.random_state = random_state
        self.classes_ = np.asarray(getattr(estimator, "classes_", [1.0, 2.0, 3.0]))
        self.class_quantiles_: Dict[float, float] = {}
        self.class_thresholds_: Dict[float, float] = {}
        self.n_cal_per_class_: Dict[float, int] = {}
        self.is_conformalized_: bool = False

    def conformalize(
        self,
        X_cal: Union[pd.DataFrame, np.ndarray],
        y_cal: Union[pd.Series, np.ndarray],
        alpha: Optional[float] = None
    ) -> "MondrianConformalClassifier":
        """
        Fit class-specific non-conformity quantiles on the calibration partition.

        Parameters
        ----------
        X_cal : calibration features
        y_cal : calibration true labels (1.0, 2.0, 3.0)
        alpha : significance level (default: self.alpha)
        """
        if alpha is not None:
            self.alpha = float(alpha)

        y_cal_arr = np.asarray(y_cal)
        probs_cal = self.estimator.predict_proba(X_cal)

        self.class_quantiles_ = {}
        self.class_thresholds_ = {}
        self.n_cal_per_class_ = {}

        for k_idx, c in enumerate(self.classes_):
            mask_c = (y_cal_arr == c)
            n_k = int(np.sum(mask_c))
            self.n_cal_per_class_[c] = n_k

            if n_k == 0:
                # Fallback if no calibration samples exist for class
                self.class_quantiles_[c] = 1.0
                self.class_thresholds_[c] = 0.0
                continue

            # Non-conformity score for true class: s_i = 1 - P(Y = c | X_i)
            scores_k = 1.0 - probs_cal[mask_c, k_idx]

            # Exact finite-sample rank order statistic: k = ceil((n_k + 1) * (1 - alpha))
            rank = int(np.ceil((n_k + 1) * (1.0 - self.alpha)))

            if rank > n_k:
                # When ceil((n+1)(1-alpha)) > n, the finite-sample guarantee cannot be met
                # without unconditionally including class c (probability threshold = 0.0)
                self.class_quantiles_[c] = 1.0
                self.class_thresholds_[c] = 0.0
            else:
                scores_sorted = np.sort(scores_k)
                # 0-indexed rank: rank - 1
                q_k = float(scores_sorted[rank - 1])
                self.class_quantiles_[c] = q_k
                self.class_thresholds_[c] = max(0.0, 1.0 - q_k)

        self.is_conformalized_ = True
        return self

    def predict_set(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Output boolean prediction sets for input samples.
        A candidate label c is included in Gamma(x) iff P(Y = c | x) >= 1 - q_c.

        Returns
        -------
        y_ps : 2D boolean array of shape (n_samples, n_classes)
        """
        if not self.is_conformalized_:
            raise RuntimeError("Classifier has not been conformalized. Call .conformalize(X_cal, y_cal) first.")

        probs = self.estimator.predict_proba(X)
        n_samples = len(probs)
        y_ps = np.zeros((n_samples, len(self.classes_)), dtype=bool)

        for k_idx, c in enumerate(self.classes_):
            thresh = self.class_thresholds_[c]
            y_ps[:, k_idx] = (probs[:, k_idx] >= thresh)

        return y_ps

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Standard point predictions (argmax from base estimator)."""
        return self.estimator.predict(X)

    def predict_label_lists(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        class_names: Optional[Dict[float, str]] = None
    ) -> List[List[str]]:
        """
        Return human-readable prediction set lists (e.g. [['Normal'], ['Suspect', 'Pathological']]).
        """
        if class_names is None:
            class_names = CLASS_NAMES_DEFAULT

        y_ps = self.predict_set(X)
        results = []
        for i in range(len(y_ps)):
            included = [class_names.get(self.classes_[k], str(self.classes_[k])) 
                        for k in range(len(self.classes_)) if y_ps[i, k]]
            results.append(included)
        return results


def evaluate_mondrian_coverage(
    y_true: Union[pd.Series, np.ndarray],
    y_ps: np.ndarray,
    classes: np.ndarray,
    class_names: Optional[Dict[float, str]] = None
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Compute per-class coverage, marginal coverage, average set size,
    and prediction set size composition (empty, singleton, multi-class).
    """
    if class_names is None:
        class_names = CLASS_NAMES_DEFAULT

    y_arr = np.asarray(y_true)
    n_samples = len(y_arr)
    set_sizes = np.sum(y_ps, axis=1)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[y] for y in y_arr])

    correct_in_set = np.array([y_ps[i, y_indices[i]] for i in range(n_samples)], dtype=bool)

    marginal_cov = float(np.mean(correct_in_set))
    avg_size = float(np.mean(set_sizes))

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


def sweep_alpha_mondrian(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    alphas: Optional[List[float]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sweep across significance levels alpha for Mondrian Conformal Prediction.
    """
    if alphas is None:
        alphas = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]

    classes = np.asarray(getattr(estimator, "classes_", [1.0, 2.0, 3.0]))
    rows = []

    for alpha in alphas:
        target_conf = round(1.0 - alpha, 4)
        mcp = MondrianConformalClassifier(estimator=estimator, alpha=alpha, random_state=random_state)
        mcp.conformalize(X_cal, y_cal)
        y_ps = mcp.predict_set(X_test)

        metrics = evaluate_mondrian_coverage(y_test, y_ps, classes=classes)

        row = {
            "method": "MONDRIAN_LAC",
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


def run_calibration_sensitivity_experiment(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    alpha: float = 0.10,
    fractions: Optional[List[float]] = None,
    n_trials: int = 30,
    random_state: int = 42
) -> pd.DataFrame:
    """
    RQ3: Calibration-Set Size Sensitivity Experiment.

    Subsamples calibration partition at varying fractions, running repeated
    Monte Carlo iterations to evaluate empirical coverage stability and
    prediction set volatility for the minority Pathological class.

    Reports: Mean +/- Std across trials (per Project Rule 6).
    """
    if fractions is None:
        fractions = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]

    classes = np.asarray(getattr(estimator, "classes_", [1.0, 2.0, 3.0]))
    y_cal_arr = np.asarray(y_cal)
    y_test_arr = np.asarray(y_test)
    probs_cal = estimator.predict_proba(X_cal)
    probs_test = estimator.predict_proba(X_test)

    summary_rows = []

    for frac in fractions:
        cal_size = int(len(y_cal_arr) * frac)
        patho_covs = []
        patho_sizes = []
        suspect_covs = []
        normal_covs = []
        overall_covs = []
        overall_sizes = []

        if frac == 1.00:
            # Bootstrap resample full calibration set to estimate variance realistically
            rng = np.random.RandomState(random_state)
            trials = [rng.choice(len(y_cal_arr), size=len(y_cal_arr), replace=True) for _ in range(n_trials)]
        else:
            sss = StratifiedShuffleSplit(n_splits=n_trials, train_size=cal_size, random_state=random_state)
            trials = [idx for idx, _ in sss.split(probs_cal, y_cal_arr)]

        for sub_idx in trials:
            sub_probs_cal = probs_cal[sub_idx]
            sub_y_cal = y_cal_arr[sub_idx]

            # Compute Mondrian class thresholds on subsample using exact rank order statistic
            thresholds = {}
            for k_idx, c in enumerate(classes):
                mask_c = (sub_y_cal == c)
                n_k = int(np.sum(mask_c))
                if n_k == 0:
                    thresholds[c] = 0.0
                    continue
                scores_k = 1.0 - sub_probs_cal[mask_c, k_idx]
                rank = int(np.ceil((n_k + 1) * (1.0 - alpha)))
                if rank > n_k:
                    thresholds[c] = 0.0
                else:
                    scores_sorted = np.sort(scores_k)
                    q_k = float(scores_sorted[rank - 1])
                    thresholds[c] = max(0.0, 1.0 - q_k)

            # Test prediction sets
            ps_test = np.zeros((len(X_test), len(classes)), dtype=bool)
            for k_idx, c in enumerate(classes):
                ps_test[:, k_idx] = (probs_test[:, k_idx] >= thresholds[c])

            # Class-specific metrics
            # Pathological (3.0)
            mask_p = (y_test_arr == 3.0)
            patho_covs.append(float(np.mean(ps_test[mask_p, 2])))
            patho_sizes.append(float(np.mean(np.sum(ps_test[mask_p], axis=1))))

            # Suspect (2.0)
            mask_s = (y_test_arr == 2.0)
            suspect_covs.append(float(np.mean(ps_test[mask_s, 1])))

            # Normal (1.0)
            mask_n = (y_test_arr == 1.0)
            normal_covs.append(float(np.mean(ps_test[mask_n, 0])))

            # Overall
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_test_idx = np.array([class_to_idx[y] for y in y_test_arr])
            correct = [ps_test[i, y_test_idx[i]] for i in range(len(y_test_arr))]
            overall_covs.append(float(np.mean(correct)))
            overall_sizes.append(float(np.mean(np.sum(ps_test, axis=1))))

        row = {
            "cal_fraction": frac,
            "n_cal_total": cal_size,
            "n_patho_cal_approx": int(np.sum(y_cal_arr == 3.0) * frac),
            "n_trials": len(trials),
            "patho_cov_mean": round(float(np.mean(patho_covs)), 4),
            "patho_cov_std": round(float(np.std(patho_covs)), 4),
            "patho_size_mean": round(float(np.mean(patho_sizes)), 4),
            "patho_size_std": round(float(np.std(patho_sizes)), 4),
            "suspect_cov_mean": round(float(np.mean(suspect_covs)), 4),
            "suspect_cov_std": round(float(np.std(suspect_covs)), 4),
            "normal_cov_mean": round(float(np.mean(normal_covs)), 4),
            "normal_cov_std": round(float(np.std(normal_covs)), 4),
            "overall_cov_mean": round(float(np.mean(overall_covs)), 4),
            "overall_cov_std": round(float(np.std(overall_covs)), 4),
            "overall_size_mean": round(float(np.mean(overall_sizes)), 4),
            "overall_size_std": round(float(np.std(overall_sizes)), 4),
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_mondrian_comparison(
    df_global_lac: pd.DataFrame,
    df_mondrian_lac: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate 4-panel comparison figure between Global LAC and Mondrian LAC:
    1. Per-Class Coverage Bar Chart at alpha=0.10 (Proving minority recovery)
    2. Empirical Coverage vs. Target Confidence (1 - alpha)
    3. Average Set Size vs. Target Confidence (Quantifying efficiency cost)
    4. Set Composition Breakdown (Singletons vs. Ambiguous Sets)
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Per-Class Coverage at alpha=0.10
    ax1 = axes[0, 0]
    lac_10 = df_global_lac[np.isclose(df_global_lac["alpha"], 0.10)].iloc[0]
    mon_10 = df_mondrian_lac[np.isclose(df_mondrian_lac["alpha"], 0.10)].iloc[0]

    classes_list = ["Normal", "Suspect", "Pathological"]
    global_covs = [lac_10["cov_Normal"], lac_10["cov_Suspect"], lac_10["cov_Pathological"]]
    mondrian_covs = [mon_10["cov_Normal"], mon_10["cov_Suspect"], mon_10["cov_Pathological"]]

    x = np.arange(len(classes_list))
    width = 0.35
    ax1.bar(x - width/2, global_covs, width, label="Global LAC", color="#1f77b4", alpha=0.85)
    ax1.bar(x + width/2, mondrian_covs, width, label="Mondrian LAC", color="#2ca02c", alpha=0.85)
    ax1.axhline(0.90, color="red", linestyle="--", linewidth=1.5, label="Target 90% Guarantee")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes_list, fontweight="bold")
    ax1.set_ylabel("Empirical Class Coverage", fontweight="bold")
    ax1.set_title("Panel A: Per-Class Coverage (Global vs. Mondrian at α = 0.10)", fontweight="bold", fontsize=11)
    ax1.set_ylim([0.0, 1.10])
    ax1.legend(loc="lower left")

    for i, v in enumerate(global_covs):
        ax1.text(i - width/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(mondrian_covs):
        ax1.text(i + width/2, v + 0.02, f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Target vs Empirical Coverage across Alphas
    ax2 = axes[0, 1]
    ax2.plot(df_global_lac["target_confidence"], df_global_lac["empirical_coverage"], marker="o", label="Global LAC", color="#1f77b4", linewidth=2)
    ax2.plot(df_mondrian_lac["target_confidence"], df_mondrian_lac["empirical_coverage"], marker="^", label="Mondrian LAC", color="#2ca02c", linewidth=2)
    min_c = min(df_global_lac["target_confidence"].min(), df_mondrian_lac["target_confidence"].min())
    max_c = max(df_global_lac["target_confidence"].max(), df_mondrian_lac["target_confidence"].max())
    ax2.plot([min_c, max_c], [min_c, max_c], "k--", label="Ideal Line (y = x)", alpha=0.7)
    ax2.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax2.set_ylabel("Empirical Coverage", fontweight="bold")
    ax2.set_title("Panel B: Overall Coverage Validity across Alphas", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper left")

    # Panel 3: Average Set Size Comparison (Efficiency)
    ax3 = axes[1, 0]
    ax3.plot(df_global_lac["target_confidence"], df_global_lac["avg_set_size"], marker="o", label="Global LAC Avg Size", color="#1f77b4", linewidth=2)
    ax3.plot(df_mondrian_lac["target_confidence"], df_mondrian_lac["avg_set_size"], marker="^", label="Mondrian LAC Avg Size", color="#2ca02c", linewidth=2)
    ax3.axhline(1.0, color="gray", linestyle=":", label="Single Label (|S|=1)")
    ax3.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax3.set_ylabel("Average Prediction Set Size", fontweight="bold")
    ax3.set_title("Panel C: Efficiency Cost (Set Size vs. Confidence)", fontweight="bold", fontsize=11)
    ax3.legend(loc="upper left")

    # Panel 4: Set Composition Breakdown for Mondrian (Categorical bars to avoid overlapping)
    ax4 = axes[1, 1]
    n_bars = len(df_mondrian_lac)
    x_indices = np.arange(n_bars)
    bar_w = 0.55
    ax4.bar(x_indices, df_mondrian_lac["empty_pct"], width=bar_w, label="Empty Set (|S|=0)", color="#d62728", alpha=0.85)
    ax4.bar(x_indices, df_mondrian_lac["singleton_pct"], width=bar_w, bottom=df_mondrian_lac["empty_pct"], label="Singleton (|S|=1)", color="#2ca02c", alpha=0.85)
    ax4.bar(x_indices, df_mondrian_lac["multi_pct"], width=bar_w, bottom=df_mondrian_lac["empty_pct"] + df_mondrian_lac["singleton_pct"], label="Ambiguous (|S|>=2)", color="#9467bd", alpha=0.85)

    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"{c:.2f}" for c in df_mondrian_lac["target_confidence"]], fontweight="bold")
    ax4.set_xlabel("Target Confidence (1 - α)", fontweight="bold")
    ax4.set_ylabel("Set Composition (%)", fontweight="bold")
    ax4.set_title("Panel D: Mondrian LAC Prediction Set Composition", fontweight="bold", fontsize=11)
    ax4.set_ylim([0, 105])
    ax4.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.conformal.mondrian_cp] Saved Mondrian comparison figure to: {save_path}")

    return fig


def plot_sensitivity_curves(
    df_sensitivity: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate 2-panel publication figure for RQ3 (Calibration-Set Size Sensitivity):
    1. Pathological Class Coverage Stability (Mean +/- 1 Std vs. Calibration Size)
    2. Pathological Prediction Set Size & Overall Size Stability (Mean +/- 1 Std)
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = df_sensitivity["n_cal_total"]

    # Panel 1: Coverage Stability for Pathological Class
    ax1 = axes[0]
    mean_cov = df_sensitivity["patho_cov_mean"]
    std_cov = df_sensitivity["patho_cov_std"]
    ax1.plot(x, mean_cov, marker="o", color="#d62728", linewidth=2.5, label="Pathological Coverage (Mean)")
    ax1.fill_between(x, mean_cov - std_cov, mean_cov + std_cov, color="#d62728", alpha=0.2, label="±1 Std Dev (Volatility)")
    ax1.axhline(0.90, color="black", linestyle="--", linewidth=1.5, label="Target 90% Guarantee")
    ax1.set_xlabel("Total Calibration Set Size (N_cal)", fontweight="bold")
    ax1.set_ylabel("Empirical Coverage on Test Set", fontweight="bold")
    ax1.set_title("Panel A: Pathological Class Coverage Stability vs. Calibration Size (RQ3)", fontweight="bold", fontsize=11)
    ax1.set_ylim([0.65, 1.05])
    ax1.legend(loc="lower right")

    # Annotate minority sample sizes
    for _, row in df_sensitivity.iterrows():
        ax1.annotate(f"n_patho≈{int(row['n_patho_cal_approx'])}",
                     (row["n_cal_total"], row["patho_cov_mean"]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    # Panel 2: Prediction Set Size Volatility
    ax2 = axes[1]
    mean_size = df_sensitivity["patho_size_mean"]
    std_size = df_sensitivity["patho_size_std"]
    overall_size = df_sensitivity["overall_size_mean"]
    overall_std = df_sensitivity["overall_size_std"]

    ax2.plot(x, mean_size, marker="s", color="#9467bd", linewidth=2.5, label="Pathological Set Size (Mean ± 1 Std)")
    ax2.fill_between(x, mean_size - std_size, mean_size + std_size, color="#9467bd", alpha=0.2)
    ax2.plot(x, overall_size, marker="^", color="#2ca02c", linewidth=2.0, linestyle="--", label="Overall Set Size (Mean ± 1 Std)")
    ax2.fill_between(x, overall_size - overall_std, overall_size + overall_std, color="#2ca02c", alpha=0.15)
    ax2.axhline(1.0, color="gray", linestyle=":", label="Single Label (|S|=1)")

    ax2.set_xlabel("Total Calibration Set Size (N_cal)", fontweight="bold")
    ax2.set_ylabel("Average Prediction Set Size", fontweight="bold")
    ax2.set_title("Panel B: Prediction Set Size Volatility vs. Calibration Size", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.conformal.mondrian_cp] Saved Sensitivity figure to: {save_path}")

    return fig
