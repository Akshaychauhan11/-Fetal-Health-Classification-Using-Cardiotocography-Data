"""
Selective Classification and Abstention Mechanism for Fetal Health Classification.

Provides:
1. Accept/Abstain Decision Rules:
   - Singleton prediction set (|S| == 1): ACCEPT
   - Empty set (|S| == 0) or Multi-class set (|S| >= 2): ABSTAIN
   - Pathological ambiguity flag: High priority alert for human obstetric review.
2. A-E Method Comparison:
   - Method A: Baseline (No Abstention / Argmax)
   - Method B: Raw Confidence Thresholding
   - Method C: Calibrated Probability Thresholding (Platt Scaling)
   - Method D: Global Conformal Prediction Abstention
   - Method E: Mondrian Conformal Prediction Abstention
3. Risk-Coverage Curve Computation (Selective Risk vs. Coverage Phi).
4. Publication-grade plotting for Risk-Coverage Curves.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Default class names
CLASS_NAMES_DEFAULT = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


def decide_prediction(
    prediction_set: List[str],
    predicted_label: Optional[str] = None,
    confidence_val: Optional[float] = None
) -> Dict[str, Union[str, List[str], float]]:
    """
    Apply selective prediction decision logic to a single patient's prediction set.

    Rules:
    - If |S| == 1: ACCEPT (decision="ACCEPT", confidence="high")
    - If |S| == 0: ABSTAIN (decision="ABSTAIN", reason="Empty prediction set / Total ambiguity")
    - If |S| >= 2: ABSTAIN (decision="ABSTAIN", reason="Multi-class clinical ambiguity")
      Special case: If 'Pathological' is inside a multi-class set, reason="High-Risk Pathological Ambiguity"

    Returns
    -------
    dict matching FastAPI contract:
        - 'decision': 'ACCEPT' | 'ABSTAIN'
        - 'prediction_set': list of class strings
        - 'predicted_label': str (the accepted diagnosis or most likely candidate)
        - 'confidence_level': 'high' | 'moderate' | 'low'
        - 'reason': str explanation
    """
    set_size = len(prediction_set)

    if set_size == 1:
        label = prediction_set[0]
        return {
            "decision": "ACCEPT",
            "prediction_set": prediction_set,
            "predicted_label": label,
            "confidence_level": "high",
            "reason": f"Definitive single diagnosis: {label}"
        }
    elif set_size == 0:
        return {
            "decision": "ABSTAIN",
            "prediction_set": [],
            "predicted_label": predicted_label if predicted_label else "Uncertain",
            "confidence_level": "low",
            "reason": "Empty prediction set (Atypical non-conforming CTG pattern)"
        }
    else:
        # Multi-class set (|S| >= 2)
        has_patho = "Pathological" in prediction_set
        reason = "High-Risk Pathological Ambiguity" if has_patho else f"Ambiguous between: {', '.join(prediction_set)}"
        return {
            "decision": "ABSTAIN",
            "prediction_set": prediction_set,
            "predicted_label": predicted_label if predicted_label else prediction_set[0],
            "confidence_level": "low",
            "reason": reason
        }


def evaluate_selective_abstention(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    accept_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate Coverage (Phi), Selective Risk (R), Selective Accuracy,
    False Reassurance count/rate (True Pathological predicted as Normal),
    and False Alarm count/rate (True Normal predicted as Pathological).

    Parameters
    ----------
    y_true : 1D array of true labels (1.0=Normal, 2.0=Suspect, 3.0=Pathological)
    y_pred : 1D array of predicted labels
    accept_mask : 1D boolean array where True = ACCEPT, False = ABSTAIN

    Returns
    -------
    dict with metrics on accepted cases
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    n_total = len(y_t)
    n_accepted = int(np.sum(accept_mask))
    n_abstained = n_total - n_accepted

    coverage = float(n_accepted / n_total)

    if n_accepted > 0:
        y_t_acc = y_t[accept_mask]
        y_p_acc = y_p[accept_mask]
        errors = int(np.sum(y_p_acc != y_t_acc))
        selective_risk = float(errors / n_accepted)
        selective_acc = float(1.0 - selective_risk)

        # False Reassurance: True Pathological (3.0) diagnosed as Normal (1.0) among accepted
        false_reassurances = int(np.sum((y_t_acc == 3.0) & (y_p_acc == 1.0)))
        total_patho_accepted = int(np.sum(y_t_acc == 3.0))

        # False Alarm: True Normal (1.0) diagnosed as Pathological (3.0) among accepted
        false_alarms = int(np.sum((y_t_acc == 1.0) & (y_p_acc == 3.0)))
        total_normal_accepted = int(np.sum(y_t_acc == 1.0))
    else:
        selective_risk = np.nan
        selective_acc = np.nan
        false_reassurances = 0
        total_patho_accepted = 0
        false_alarms = 0
        total_normal_accepted = 0

    return {
        "coverage": round(coverage, 4),
        "selective_risk": round(selective_risk, 4) if not np.isnan(selective_risk) else np.nan,
        "selective_accuracy": round(selective_acc, 4) if not np.isnan(selective_acc) else np.nan,
        "n_accepted": n_accepted,
        "n_abstained": n_abstained,
        "false_reassurances": false_reassurances,
        "total_patho_accepted": total_patho_accepted,
        "false_alarms": false_alarms,
        "total_normal_accepted": total_normal_accepted,
    }


def compute_ae_method_comparison(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    alpha: float = 0.10,
    matched_coverage: Optional[float] = None
) -> pd.DataFrame:
    """
    Build the A-E Method Comparison Table on the test set.
    Computes BOTH RF-argmax risk (standard definition where accepted cases use the base model argmax)
    and singleton-label risk for conformal methods, alongside false reassurances and false alarms.

    If matched_coverage is provided, Methods B and C thresholds are tuned on the calibration set
    to match that target coverage (eliminating test-set snooping); otherwise fixed thresholds (tau=0.80) are used.
    """
    from src.conformal.global_cp import fit_global_conformal_classifier, predict_prediction_sets
    from src.conformal.mondrian_cp import MondrianConformalClassifier

    classes = np.asarray(getattr(estimator, "classes_", [1.0, 2.0, 3.0]))
    y_cal_arr = np.asarray(y_cal)
    missing_cal_classes = np.setdiff1d(classes, np.unique(y_cal_arr))
    if len(missing_cal_classes) > 0:
        raise ValueError(f"Calibration split is missing classes present in estimator: {missing_cal_classes}")

    y_test_arr = np.asarray(y_test)
    raw_probs = estimator.predict_proba(X_test)
    raw_preds = classes[np.argmax(raw_probs, axis=1)]
    raw_conf = np.max(raw_probs, axis=1)

    # Conformal sets first so we can match coverage if requested
    # Method D: Global Conformal LAC
    cp_lac = fit_global_conformal_classifier(estimator, X_cal, y_cal, method="lac", confidence_level=1.0 - alpha)
    _, y_ps_d = predict_prediction_sets(cp_lac, X_test)
    if y_ps_d.ndim == 3:
        y_ps_d = y_ps_d[:, :, 0]
    sizes_d = np.sum(y_ps_d, axis=1)
    mask_d = (sizes_d == 1)
    preds_d_single = np.array([classes[np.argmax(y_ps_d[i])] if mask_d[i] else -1 for i in range(len(y_test_arr))])

    # Method E: Mondrian Conformal LAC
    mcp = MondrianConformalClassifier(estimator, alpha=alpha).conformalize(X_cal, y_cal)
    y_ps_e = mcp.predict_set(X_test)
    sizes_e = np.sum(y_ps_e, axis=1)
    mask_e = (sizes_e == 1)
    mcp_classes = np.asarray(getattr(mcp, "classes_", classes))
    preds_e_single = np.array([mcp_classes[np.argmax(y_ps_e[i])] if mask_e[i] else -1 for i in range(len(y_test_arr))])

    # Calibrator for Method C: Platt Scaling
    cal_probs_cal_input = estimator.predict_proba(X_cal)
    cal_lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    cal_lr.fit(cal_probs_cal_input, y_cal)
    cal_classes = np.asarray(getattr(cal_lr, "classes_", classes))
    cal_probs = cal_lr.predict_proba(raw_probs)
    cal_conf = np.max(cal_probs, axis=1)
    cal_preds = cal_classes[np.argmax(cal_probs, axis=1)]

    # Determine thresholds for B and C
    taus = np.linspace(0.40, 0.99, 500)
    if matched_coverage is not None:
        target_cov = matched_coverage
        # Choose and freeze cutoffs strictly on calibration split to eliminate test-set snooping
        raw_conf_cal = np.max(cal_probs_cal_input, axis=1)
        cal_conf_cal = np.max(cal_lr.predict_proba(cal_probs_cal_input), axis=1)
        tau_b = float(taus[np.argmin([abs(np.mean(raw_conf_cal >= t) - target_cov) for t in taus])])
        tau_c = float(taus[np.argmin([abs(np.mean(cal_conf_cal >= t) - target_cov) for t in taus])])
        crit_b = f"max P_raw >= {tau_b:.3f} (cal-tuned to target {target_cov:.1%})"
        crit_c = f"max P_calib >= {tau_c:.3f} (cal-tuned to target {target_cov:.1%})"
    else:
        tau_b = 0.80
        tau_c = 0.80
        crit_b = "max P_raw >= 0.80 (fixed)"
        crit_c = "max P_calib >= 0.80 (fixed)"

    mask_b = (raw_conf >= tau_b)
    mask_c = (cal_conf >= tau_c)

    rows = []

    # Method A: Baseline
    mask_a = np.ones(len(y_test_arr), dtype=bool)
    m_a = evaluate_selective_abstention(y_test_arr, raw_preds, mask_a)
    rows.append({
        "Method": "Method A: Baseline (No Abstention)",
        "Criterion": "Forced RF argmax",
        "Coverage": m_a["coverage"],
        "Selective Risk (RF argmax)": m_a["selective_risk"],
        "Selective Accuracy": m_a["selective_accuracy"],
        "Abstained Count": m_a["n_abstained"],
        "False Reassurances": m_a["false_reassurances"],
        "False Alarms": m_a["false_alarms"]
    })

    # Method B
    m_b = evaluate_selective_abstention(y_test_arr, raw_preds, mask_b)
    rows.append({
        "Method": f"Method B: Raw Conf Threshold",
        "Criterion": crit_b,
        "Coverage": m_b["coverage"],
        "Selective Risk (RF argmax)": m_b["selective_risk"],
        "Selective Accuracy": m_b["selective_accuracy"],
        "Abstained Count": m_b["n_abstained"],
        "False Reassurances": m_b["false_reassurances"],
        "False Alarms": m_b["false_alarms"]
    })

    # Method C
    # Evaluate raw RF argmax risk on accepted mask_c AND calibrated prediction risk
    m_c_rf = evaluate_selective_abstention(y_test_arr, raw_preds, mask_c)
    m_c_cal = evaluate_selective_abstention(y_test_arr, cal_preds, mask_c)
    rows.append({
        "Method": f"Method C: Calibrated Threshold (Platt)",
        "Criterion": crit_c,
        "Coverage": m_c_rf["coverage"],
        "Selective Risk (RF argmax)": m_c_rf["selective_risk"],
        "Selective Risk (Calibrated)": m_c_cal["selective_risk"],
        "Selective Accuracy": m_c_rf["selective_accuracy"],
        "Abstained Count": m_c_rf["n_abstained"],
        "False Reassurances": m_c_rf["false_reassurances"],
        "False Alarms": m_c_rf["false_alarms"]
    })

    # Method D: Global Conformal
    # Evaluated with RF argmax on accepted cases and singleton prediction
    m_d_argmax = evaluate_selective_abstention(y_test_arr, raw_preds, mask_d)
    m_d_single = evaluate_selective_abstention(y_test_arr, preds_d_single, mask_d)
    rows.append({
        "Method": f"Method D: Global Conformal LAC (α={alpha:.2f})",
        "Criterion": "|S_global| == 1 (Singleton)",
        "Coverage": m_d_argmax["coverage"],
        "Selective Risk (RF argmax)": m_d_argmax["selective_risk"],
        "Selective Accuracy": m_d_argmax["selective_accuracy"],
        "Abstained Count": m_d_argmax["n_abstained"],
        "False Reassurances": m_d_argmax["false_reassurances"],
        "False Alarms": m_d_argmax["false_alarms"],
        "Singleton-Only Risk": m_d_single["selective_risk"],
        "False Reassurances (Singleton)": m_d_single["false_reassurances"],
        "False Alarms (Singleton)": m_d_single["false_alarms"]
    })

    # Method E: Mondrian Conformal
    m_e_argmax = evaluate_selective_abstention(y_test_arr, raw_preds, mask_e)
    m_e_single = evaluate_selective_abstention(y_test_arr, preds_e_single, mask_e)
    rows.append({
        "Method": f"Method E: Mondrian Conformal LAC (α={alpha:.2f})",
        "Criterion": "|S_mondrian| == 1 (Singleton)",
        "Coverage": m_e_argmax["coverage"],
        "Selective Risk (RF argmax)": m_e_argmax["selective_risk"],
        "Selective Accuracy": m_e_argmax["selective_accuracy"],
        "Abstained Count": m_e_argmax["n_abstained"],
        "False Reassurances": m_e_argmax["false_reassurances"],
        "False Alarms": m_e_argmax["false_alarms"],
        "Singleton-Only Risk": m_e_single["selective_risk"],
        "False Reassurances (Singleton)": m_e_single["false_reassurances"],
        "False Alarms (Singleton)": m_e_single["false_alarms"]
    })

    return pd.DataFrame(rows)


def compute_risk_coverage_curves(
    estimator,
    X_cal: Union[pd.DataFrame, np.ndarray],
    y_cal: Union[pd.Series, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray]
) -> Dict[str, pd.DataFrame]:
    """
    Generate points along the Risk-Coverage (RC) curve for Methods B, C, D, and E.
    Returns schema-bearing DataFrames even when sweeps have no qualifying points.
    """
    from src.conformal.global_cp import fit_global_conformal_classifier, predict_prediction_sets
    from src.conformal.mondrian_cp import MondrianConformalClassifier

    classes = np.asarray(getattr(estimator, "classes_", [1.0, 2.0, 3.0]))
    y_test_arr = np.asarray(y_test)
    raw_probs = estimator.predict_proba(X_test)
    raw_conf = np.max(raw_probs, axis=1)
    raw_preds = classes[np.argmax(raw_probs, axis=1)]

    # Thresholds for confidence methods
    thresholds = np.linspace(0.35, 0.99, 45)

    # 1. Method B: Raw Confidence Thresholding
    rc_raw = []
    for tau in thresholds:
        mask = (raw_conf >= tau)
        if np.sum(mask) >= 5:
            res = evaluate_selective_abstention(y_test_arr, raw_preds, mask)
            rc_raw.append({"coverage": res["coverage"], "risk": res["selective_risk"], "tau": tau})
    df_rc_raw = pd.DataFrame(rc_raw, columns=["coverage", "risk", "tau"])
    if not df_rc_raw.empty:
        df_rc_raw = df_rc_raw.drop_duplicates(subset=["coverage"]).sort_values("coverage")

    # 2. Method C: Calibrated Probability Thresholding (Platt Scaling)
    cal_lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    cal_lr.fit(estimator.predict_proba(X_cal), y_cal)
    cal_probs = cal_lr.predict_proba(raw_probs)
    cal_conf = np.max(cal_probs, axis=1)
    cal_classes = np.asarray(getattr(cal_lr, "classes_", classes))
    cal_preds = cal_classes[np.argmax(cal_probs, axis=1)]

    rc_cal = []
    for tau in thresholds:
        mask = (cal_conf >= tau)
        if np.sum(mask) >= 5:
            res = evaluate_selective_abstention(y_test_arr, cal_preds, mask)
            rc_cal.append({"coverage": res["coverage"], "risk": res["selective_risk"], "tau": tau})
    df_rc_cal = pd.DataFrame(rc_cal, columns=["coverage", "risk", "tau"])
    if not df_rc_cal.empty:
        df_rc_cal = df_rc_cal.drop_duplicates(subset=["coverage"]).sort_values("coverage")

    # 3. Method D: Global Conformal Sweep
    alphas = np.linspace(0.01, 0.40, 35)
    rc_global = []
    for a in alphas:
        cp = fit_global_conformal_classifier(estimator, X_cal, y_cal, method="lac", confidence_level=1.0 - a)
        _, y_ps = predict_prediction_sets(cp, X_test)
        if y_ps.ndim == 3:
            y_ps = y_ps[:, :, 0]
        sizes = np.sum(y_ps, axis=1)
        mask = (sizes == 1)
        if np.sum(mask) >= 5:
            preds = np.array([classes[np.argmax(y_ps[i])] if mask[i] else -1 for i in range(len(y_test_arr))])
            res = evaluate_selective_abstention(y_test_arr, preds, mask)
            rc_global.append({"coverage": res["coverage"], "risk": res["selective_risk"], "alpha": a})
    df_rc_global = pd.DataFrame(rc_global, columns=["coverage", "risk", "alpha"])
    if not df_rc_global.empty:
        df_rc_global = df_rc_global.drop_duplicates(subset=["coverage"]).sort_values("coverage")

    # 4. Method E: Mondrian Conformal Sweep
    rc_mondrian = []
    for a in alphas:
        mcp = MondrianConformalClassifier(estimator, alpha=a).conformalize(X_cal, y_cal)
        y_ps = mcp.predict_set(X_test)
        sizes = np.sum(y_ps, axis=1)
        mask = (sizes == 1)
        if np.sum(mask) >= 5:
            mcp_classes = np.asarray(getattr(mcp, "classes_", classes))
            preds = np.array([mcp_classes[np.argmax(y_ps[i])] if mask[i] else -1 for i in range(len(y_test_arr))])
            res = evaluate_selective_abstention(y_test_arr, preds, mask)
            rc_mondrian.append({"coverage": res["coverage"], "risk": res["selective_risk"], "alpha": a})
    df_rc_mondrian = pd.DataFrame(rc_mondrian, columns=["coverage", "risk", "alpha"])
    if not df_rc_mondrian.empty:
        df_rc_mondrian = df_rc_mondrian.drop_duplicates(subset=["coverage"]).sort_values("coverage")

    return {
        "raw": df_rc_raw,
        "calibrated": df_rc_cal,
        "global_cp": df_rc_global,
        "mondrian_cp": df_rc_mondrian
    }


def plot_risk_coverage_curves(
    rc_curves_dict: Dict[str, pd.DataFrame],
    baseline_risk: float,
    df_ae_comparison: Optional[pd.DataFrame] = None,
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Generate 2-panel publication figure for Phase 4:
    Panel A: Risk-Coverage Curves comparing Methods B, C, D, E against Baseline A.
    Panel B: Bar chart of Selective Accuracy and Abstained Case Count for Methods A-E.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Risk-Coverage Curves
    ax1 = axes[0]
    df_raw = rc_curves_dict["raw"]
    df_cal = rc_curves_dict["calibrated"]
    df_global = rc_curves_dict["global_cp"]
    df_mon = rc_curves_dict["mondrian_cp"]

    # Baseline Method A point
    ax1.plot(1.0, baseline_risk, marker="*", markersize=14, color="black", label=f"Method A: Baseline (Risk={baseline_risk:.2%})", zorder=5)

    # Method B
    if not df_raw.empty and "coverage" in df_raw.columns:
        ax1.plot(df_raw["coverage"], df_raw["risk"], marker="o", markersize=4, color="#1f77b4", linewidth=2, label="Method B: Raw Conf Threshold")

    # Method C
    if not df_cal.empty and "coverage" in df_cal.columns:
        ax1.plot(df_cal["coverage"], df_cal["risk"], marker="s", markersize=4, color="#ff7f0e", linewidth=2, label="Method C: Calibrated Threshold")

    # Method D
    if not df_global.empty and "coverage" in df_global.columns:
        ax1.plot(df_global["coverage"], df_global["risk"], marker="^", markersize=5, color="#9467bd", linewidth=2, label="Method D: Global Conformal LAC")

    # Method E
    if not df_mon.empty and "coverage" in df_mon.columns:
        ax1.plot(df_mon["coverage"], df_mon["risk"], marker="d", markersize=5, color="#2ca02c", linewidth=2.5, label="Method E: Mondrian Conformal LAC")

    ax1.set_xlabel("Coverage Φ (Fraction of Accepted Cases)", fontweight="bold")
    ax1.set_ylabel("Selective Risk R (Error Rate on Accepted)", fontweight="bold")
    ax1.set_title("Panel A: Risk–Coverage Curves (Trade-off Comparison)", fontweight="bold", fontsize=11)

    # Derive dynamic limits to avoid clipping valid sweep points
    all_covs = []
    all_risks = [baseline_risk]
    for df in [df_raw, df_cal, df_global, df_mon]:
        if not df.empty and "coverage" in df.columns:
            all_covs.extend(df["coverage"].dropna().tolist())
            all_risks.extend(df["risk"].dropna().tolist())

    if all_covs:
        min_cov = max(0.0, float(np.min(all_covs)) - 0.05)
        max_cov = min(1.05, float(np.max(all_covs)) + 0.02)
        ax1.set_xlim([min_cov, max_cov])
    else:
        ax1.set_xlim([0.65, 1.02])

    if all_risks:
        max_r = max(0.08, float(np.max(all_risks)) + 0.01)
        ax1.set_ylim([-0.005, min(1.0, max_r)])
    else:
        ax1.set_ylim([-0.005, 0.08])

    ax1.legend(loc="upper left")

    # Panel B: Method A-E Comparison Bar Chart
    ax2 = axes[1]
    if df_ae_comparison is not None:
        methods = ["Method A\nBaseline", "Method B\nRaw Conf", "Method C\nCalib Conf", "Method D\nGlobal CP", "Method E\nMondrian CP"]
        accuracies = df_ae_comparison["Selective Accuracy"].tolist()
        abstained = df_ae_comparison["Abstained Count"].tolist()

        x = np.arange(len(methods))
        width = 0.45
        bars = ax2.bar(x, accuracies, width, color=["#7f7f7f", "#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontweight="bold", fontsize=9)
        ax2.set_ylabel("Selective Accuracy (1 - Risk)", fontweight="bold")
        ax2.set_title("Panel B: Accuracy on Accepted Cases (with Abstained Counts)", fontweight="bold", fontsize=11)
        
        valid_accs = [a for a in accuracies if not np.isnan(a)]
        min_acc = max(0.0, min(valid_accs) - 0.05) if valid_accs else 0.85
        ax2.set_ylim([min_acc, 1.02])

        for i, bar in enumerate(bars):
            h = bar.get_height()
            if not np.isnan(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.1%}\n(Abs: {abstained[i]})",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")
            else:
                ax2.text(bar.get_x() + bar.get_width() / 2, min_acc + 0.02, "N/A\n(0 acc)",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[src.abstention.selective] Saved Risk-Coverage plot to: {save_path}")

    return fig
