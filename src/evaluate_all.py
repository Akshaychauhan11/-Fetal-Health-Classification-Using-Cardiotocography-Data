"""
Master Evaluation and Synthesis Script: Fetal Health Uncertainty
================================================================
This script is the single authoritative source of truth for all quantitative
results, tables, figure generation, and array exports in the project.

Key Features & Strict Methodological Fixes:
1. Exact finite-sample rank-order statistics for Mondrian CP:
   rank = ceil((n_k + 1) * (1 - alpha)). If rank > n_k, threshold = 0.0.
   Otherwise, threshold = max(0.0, 1.0 - sorted_scores[rank - 1]).
2. Dual reporting of Selective Risk:
   - RF Argmax Selective Risk: Risk evaluated using the base RF model's point predictions
     on the accepted subset (standard selective classification definition).
   - Singleton-Only Risk: Conformal prediction set singleton diagnosis risk.
   - Clinical error tracking: False Reassurances (Pathological -> Normal) and False Alarms (Normal -> Pathological).
3. Matched Coverage & Fixed Threshold dual reporting:
   - Evaluates Methods B and C at both fixed tau=0.80 and calibration-matched coverage to Method E.
4. Single-split sacred test set evaluation (N=426, 35 Pathological, 59 Suspect, 332 Normal)
   AND 100 Repeated Stratified 60/20/20 Splits (Reporting Mean +/- Std).
5. RQ3 Calibration-size Sensitivity Analysis across subsampled calibration fractions (15%–75%)
   and the complete calibration partition at 100%.
6. Exports test prediction arrays (CSV and NPZ) and regenerates core uncertainty & conformal publication figures (Figs 2, 3, 4, 5, and 5B).
"""

import os
import sys

# Set headless matplotlib backend before importing pyplot or seaborn
import matplotlib
matplotlib.use("Agg")

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, confusion_matrix
import joblib

# Ensure repo root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.conformal.global_cp import (
    fit_global_conformal_classifier,
    predict_prediction_sets,
    evaluate_conformal_coverage,
    sweep_alpha_conformal,
    plot_global_conformal_comparison
)
from src.conformal.mondrian_cp import (
    MondrianConformalClassifier,
    evaluate_mondrian_coverage,
    sweep_alpha_mondrian,
    run_calibration_sensitivity_experiment,
    plot_mondrian_comparison,
    plot_sensitivity_curves
)
from src.abstention.selective import (
    evaluate_selective_abstention,
    compute_ae_method_comparison,
    compute_risk_coverage_curves,
    plot_risk_coverage_curves
)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fetal_health.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
ALPHA_DEFAULT = 0.10

CLASS_MAP = {1.0: "Normal", 2.0: "Suspect", 3.0: "Pathological"}


def load_and_split_data():
    """Load raw dataset and reproduce exact sacred 60/20/20 stratified split."""
    df = pd.read_csv(RAW_DATA_PATH)
    feature_cols = [c for c in df.columns if c != "fetal_health"]
    X = df[feature_cols]
    y = df["fetal_health"]

    # 60% Train, 40% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    # Temp -> 20% Cal, 20% Test (split temp 50/50)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    return df, X_train, y_train, X_cal, y_cal, X_test, y_test


def evaluate_baseline_model(rf, X_test, y_test):
    """Compute per-class baseline metrics and exact confusion matrix."""
    y_test_arr = np.asarray(y_test)
    probs = rf.predict_proba(X_test)
    preds = rf.predict(X_test)

    recalls = recall_score(y_test_arr, preds, average=None, labels=[1.0, 2.0, 3.0])
    acc = accuracy_score(y_test_arr, preds)
    f1_macro = f1_score(y_test_arr, preds, average="macro")
    cm = confusion_matrix(y_test_arr, preds, labels=[1.0, 2.0, 3.0])

    # Errors
    is_error = (preds != y_test_arr)
    n_errors = int(np.sum(is_error))
    base_error_rate = float(n_errors / len(y_test_arr))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "recall_Normal": recalls[0],
        "recall_Suspect": recalls[1],
        "recall_Pathological": recalls[2],
        "confusion_matrix": cm,
        "n_errors": n_errors,
        "error_rate": base_error_rate,
        "probs": probs,
        "preds": preds
    }


def run_full_evaluation():
    print("=" * 80)
    print("RUNNING MASTER EVALUATION PIPELINE FOR FETAL HEALTH UNCERTAINTY")
    print("=" * 80)

    # 1. Load Data
    df, X_train, y_train, X_cal, y_cal, X_test, y_test = load_and_split_data()
    print(f"Dataset Loaded: {len(df)} total records.")
    print(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")
    y_test_arr = np.asarray(y_test)
    print(f"Test Class Distribution: Normal={np.sum(y_test_arr==1.0)}, Suspect={np.sum(y_test_arr==2.0)}, Pathological={np.sum(y_test_arr==3.0)}")

    # 2. Train or Load Base Model (RandomForest)
    model_path = MODELS_DIR / "random_forest.joblib"
    if model_path.exists():
        rf = joblib.load(model_path)
        print("Loaded pre-trained Random Forest model.")
    else:
        print("Training Random Forest model (300 estimators, class_weight='balanced', random_state=42)...")
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train, y_train)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_model_path = model_path.with_suffix(".tmp")
        joblib.dump(rf, tmp_model_path)
        os.replace(tmp_model_path, model_path)
        print(f"Saved model atomically to {model_path}")

    # 3. Baseline Performance
    base_res = evaluate_baseline_model(rf, X_test, y_test)
    print("\n--- BASELINE RANDOM FOREST PERFORMANCE ---")
    print(f"Overall Accuracy: {base_res['accuracy']:.4f}")
    print(f"Macro F1: {base_res['f1_macro']:.4f}")
    print(f"Recall Normal (1.0): {base_res['recall_Normal']:.4f}")
    print(f"Recall Suspect (2.0): {base_res['recall_Suspect']:.4f}")
    print(f"Recall Pathological (3.0): {base_res['recall_Pathological']:.4f}")
    print(f"Baseline Errors: {base_res['n_errors']}/{len(y_test_arr)} ({base_res['error_rate']:.4%})")

    # 4. Global Conformal Prediction (Method D)
    print("\n--- EVALUATING GLOBAL CONFORMAL PREDICTION (LAC & APS) ---")
    df_global_lac = sweep_alpha_conformal(rf, X_cal, y_cal, X_test, y_test, method="lac")
    df_global_aps = sweep_alpha_conformal(rf, X_cal, y_cal, X_test, y_test, method="aps")

    # Global LAC at alpha=0.10
    cp_lac = fit_global_conformal_classifier(rf, X_cal, y_cal, method="lac", confidence_level=0.90)
    _, y_ps_d = predict_prediction_sets(cp_lac, X_test)
    if y_ps_d.ndim == 3:
        y_ps_d = y_ps_d[:, :, 0]
    global_eval = evaluate_conformal_coverage(y_test, y_ps_d, classes=rf.classes_)
    print("Global LAC (alpha=0.10) Test Metrics:")
    print(f"  Marginal Coverage: {global_eval['marginal_coverage']:.4f}")
    print(f"  Per-class Coverage: {global_eval['per_class_coverage']}")
    print(f"  Avg Set Size: {global_eval['avg_set_size']:.4f}")
    print(f"  Set Composition: Empty={global_eval['empty_set_pct']}%, Single={global_eval['singleton_set_pct']}%, Multi={global_eval['multi_class_set_pct']}%")

    # 5. Mondrian Conformal Prediction (Method E)
    print("\n--- EVALUATING MONDRIAN CONFORMAL PREDICTION (EXACT RANK STATISTIC) ---")
    df_mondrian_lac = sweep_alpha_mondrian(rf, X_cal, y_cal, X_test, y_test)
    mcp = MondrianConformalClassifier(rf, alpha=0.10, random_state=RANDOM_STATE)
    mcp.conformalize(X_cal, y_cal)
    y_ps_e = mcp.predict_set(X_test)
    mondrian_eval = evaluate_mondrian_coverage(y_test, y_ps_e, classes=rf.classes_)
    print("Mondrian LAC (alpha=0.10) Test Metrics:")
    print(f"  Marginal Coverage: {mondrian_eval['marginal_coverage']:.4f}")
    print(f"  Per-class Coverage: {mondrian_eval['per_class_coverage']}")
    print(f"  Avg Set Size: {mondrian_eval['avg_set_size']:.4f}")
    print(f"  Set Composition: Empty={mondrian_eval['empty_set_pct']}%, Single={mondrian_eval['singleton_set_pct']}%, Multi={mondrian_eval['multi_class_set_pct']}%")
    print("Mondrian Thresholds:", mcp.class_thresholds_)

    # 6. Abstention Audit on Sacred Test Set
    # Method E Singleton Mask
    sizes_e = np.sum(y_ps_e, axis=1)
    mask_e_accept = (sizes_e == 1)
    mask_e_abstain = ~mask_e_accept
    n_abstain_e = int(np.sum(mask_e_abstain))
    errors_in_abstain = int(np.sum((base_res['preds'] != y_test_arr)[mask_e_abstain]))
    abstain_error_rate = float(errors_in_abstain / n_abstain_e) if n_abstain_e > 0 else 0.0
    enrichment_factor = float(abstain_error_rate / base_res['error_rate']) if base_res['error_rate'] > 0 else 0.0

    # Fisher exact test for enrichment
    # Contingency table:
    #                 Abstained   Accepted
    # RF Error           a           b
    # RF Correct         c           d
    a = errors_in_abstain
    b = base_res['n_errors'] - a
    c = n_abstain_e - a
    d = int(np.sum(mask_e_accept)) - b
    odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d]])

    print("\n--- ABSTENTION AUDIT (METHOD E) ---")
    print(f"Abstained cases: {n_abstain_e}/{len(y_test_arr)} ({n_abstain_e/len(y_test_arr):.2%})")
    print(f"Errors inside abstention set: {a}/{n_abstain_e} ({abstain_error_rate:.2%})")
    print(f"Baseline error rate: {base_res['error_rate']:.2%}")
    print(f"Enrichment: {enrichment_factor:.2f}x (Fisher exact p = {p_val:.5f}, odds-ratio = {odds_ratio:.2f})")

    # 7. Methods A-E Comparison on Sacred Test Split
    print("\n--- METHODS A-E COMPARISON ON SACRED SPLIT ---")
    # A) Fixed tau = 0.80 for B & C
    df_ae_fixed = compute_ae_method_comparison(rf, X_cal, y_cal, X_test, y_test, alpha=0.10, matched_coverage=None)
    # B) Matched Coverage to Method E's calibration-set singleton acceptance rate (no test snooping)
    y_ps_e_cal = mcp.predict_set(X_cal)
    coverage_target_e_cal = float(np.mean(np.sum(y_ps_e_cal, axis=1) == 1))
    coverage_target_e_test = float(np.mean(mask_e_accept))
    df_ae_matched = compute_ae_method_comparison(rf, X_cal, y_cal, X_test, y_test, alpha=0.10, matched_coverage=coverage_target_e_cal)

    print("\nTable: Methods A-E with Fixed Tau (0.80):")
    print(df_ae_fixed.to_string(index=False))

    print(f"\nTable: Methods A-E with Matched Coverage (Cal target = {coverage_target_e_cal:.2%}, Method E test coverage = {coverage_target_e_test:.2%}):")
    print(df_ae_matched.to_string(index=False))

    # 8. Export Single-Split Prediction Arrays
    print("\n--- EXPORTING TEST PREDICTION ARRAYS ---")
    raw_probs_cal = rf.predict_proba(X_cal)
    raw_conf_cal = np.max(raw_probs_cal, axis=1)
    raw_probs = base_res["probs"]
    raw_conf = np.max(raw_probs, axis=1)

    # Calibrator
    cal_lr = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    cal_lr.fit(raw_probs_cal, y_cal)
    cal_probs_cal = cal_lr.predict_proba(raw_probs_cal)
    cal_conf_cal = np.max(cal_probs_cal, axis=1)
    cal_probs = cal_lr.predict_proba(raw_probs)
    cal_conf = np.max(cal_probs, axis=1)
    cal_preds = cal_lr.classes_[np.argmax(cal_probs, axis=1)]

    # Fixed threshold masks
    mask_b_fixed = (raw_conf >= 0.80)
    mask_c_fixed = (cal_conf >= 0.80)

    # Matched threshold masks tuned strictly on Calibration Set
    taus = np.linspace(0.40, 0.99, 500)
    tau_b_match = taus[np.argmin([abs(np.mean(raw_conf_cal >= t) - coverage_target_e_cal) for t in taus])]
    tau_c_match = taus[np.argmin([abs(np.mean(cal_conf_cal >= t) - coverage_target_e_cal) for t in taus])]
    mask_b_match = (raw_conf >= tau_b_match)
    mask_c_match = (cal_conf >= tau_c_match)

    sizes_d = np.sum(y_ps_d, axis=1)
    mask_d_accept = (sizes_d == 1)

    # DataFrame for CSV export
    df_arrays = pd.DataFrame({
        "sample_idx": np.arange(len(y_test_arr)),
        "true_label": y_test_arr,
        "true_label_name": [CLASS_MAP[y] for y in y_test_arr],
        "rf_pred": base_res["preds"],
        "rf_pred_name": [CLASS_MAP[p] for p in base_res["preds"]],
        "rf_prob_normal": raw_probs[:, 0],
        "rf_prob_suspect": raw_probs[:, 1],
        "rf_prob_pathological": raw_probs[:, 2],
        "rf_max_conf": raw_conf,
        "calib_prob_normal": cal_probs[:, 0],
        "calib_prob_suspect": cal_probs[:, 1],
        "calib_prob_pathological": cal_probs[:, 2],
        "calib_max_conf": cal_conf,
        "method_b_fixed_accept": mask_b_fixed,
        "method_b_matched_accept": mask_b_match,
        "method_c_fixed_accept": mask_c_fixed,
        "method_c_matched_accept": mask_c_match,
        "method_d_global_set_size": sizes_d,
        "method_d_accept": mask_d_accept,
        "method_d_in_set_Normal": y_ps_d[:, 0],
        "method_d_in_set_Suspect": y_ps_d[:, 1],
        "method_d_in_set_Patho": y_ps_d[:, 2],
        "method_e_mondrian_set_size": sizes_e,
        "method_e_accept": mask_e_accept,
        "method_e_in_set_Normal": y_ps_e[:, 0],
        "method_e_in_set_Suspect": y_ps_e[:, 1],
        "method_e_in_set_Patho": y_ps_e[:, 2],
    })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = PROCESSED_DIR / "test_predictions_comparison.csv"
    df_arrays.to_csv(csv_out, index=False)
    print(f"Saved test prediction CSV to: {csv_out}")

    npz_out = PROCESSED_DIR / "test_prediction_arrays.npz"
    np.savez_compressed(
        npz_out,
        y_true=y_test_arr,
        rf_probs=raw_probs,
        rf_preds=base_res["preds"],
        cal_probs=cal_probs,
        cal_preds=cal_preds,
        y_ps_global_lac=y_ps_d,
        y_ps_mondrian_lac=y_ps_e,
        mask_b_fixed=mask_b_fixed,
        mask_c_fixed=mask_c_fixed,
        mask_b_match=mask_b_match,
        mask_c_match=mask_c_match,
        mask_d=mask_d_accept,
        mask_e=mask_e_accept
    )
    print(f"Saved test prediction NPZ to: {npz_out}")

    # 9. 100 Repeated Stratified Splits (Evaluation across 100 random 60/20/20 splits)
    print("\n--- RUNNING 100 REPEATED STRATIFIED SPLITS (60/20/20) ---")
    N_REPEATS = 100
    feature_cols = [c for c in df.columns if c != "fetal_health"]
    X_full = df[feature_cols].values
    y_full = df["fetal_health"].values

    stats_accumulator = {
        "baseline_acc": [],
        "baseline_rec_norm": [],
        "baseline_rec_susp": [],
        "baseline_rec_path": [],
        "global_cov_marg": [],
        "global_cov_norm": [],
        "global_cov_susp": [],
        "global_cov_path": [],
        "global_size": [],
        "mondrian_cov_marg": [],
        "mondrian_cov_norm": [],
        "mondrian_cov_susp": [],
        "mondrian_cov_path": [],
        "mondrian_size": [],
        "ae_cov_A": [], "ae_risk_A": [],
        "ae_cov_B_fixed": [], "ae_risk_B_fixed": [],
        "ae_cov_B_match": [], "ae_risk_B_match": [],
        "ae_cov_C_fixed": [], "ae_risk_C_fixed": [],
        "ae_cov_C_match": [], "ae_risk_C_match": [],
        "ae_cov_D": [], "ae_risk_D_argmax": [], "ae_risk_D_single": [],
        "ae_cov_E": [], "ae_risk_E_argmax": [], "ae_risk_E_single": [],
        "enrichment_E": []
    }

    rng = np.random.RandomState(RANDOM_STATE)
    for rep in range(N_REPEATS):
        seed = int(rng.randint(0, 100000))
        # 60 train, 40 temp
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_full, y_full, test_size=0.40, random_state=seed, stratify=y_full)
        # 20 cal, 20 test
        X_c, X_te, y_c, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)

        # Base RF
        rf_rep = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf_rep.fit(X_tr, y_tr)

        # Baseline
        probs_te = rf_rep.predict_proba(X_te)
        preds_te = rf_rep.classes_[np.argmax(probs_te, axis=1)]
        conf_te = np.max(probs_te, axis=1)

        acc_rep = accuracy_score(y_te, preds_te)
        recs_rep = recall_score(y_te, preds_te, average=None, labels=[1.0, 2.0, 3.0])
        stats_accumulator["baseline_acc"].append(acc_rep)
        stats_accumulator["baseline_rec_norm"].append(recs_rep[0])
        stats_accumulator["baseline_rec_susp"].append(recs_rep[1])
        stats_accumulator["baseline_rec_path"].append(recs_rep[2])

        # Global LAC
        cp_lac_rep = fit_global_conformal_classifier(rf_rep, X_c, y_c, method="lac", confidence_level=0.90)
        _, y_ps_d_rep = predict_prediction_sets(cp_lac_rep, X_te)
        if y_ps_d_rep.ndim == 3:
            y_ps_d_rep = y_ps_d_rep[:, :, 0]
        res_d = evaluate_conformal_coverage(y_te, y_ps_d_rep, classes=rf_rep.classes_)
        stats_accumulator["global_cov_marg"].append(res_d["marginal_coverage"])
        stats_accumulator["global_cov_norm"].append(res_d["per_class_coverage"]["Normal"])
        stats_accumulator["global_cov_susp"].append(res_d["per_class_coverage"]["Suspect"])
        stats_accumulator["global_cov_path"].append(res_d["per_class_coverage"]["Pathological"])
        stats_accumulator["global_size"].append(res_d["avg_set_size"])

        # Mondrian LAC
        mcp_rep = MondrianConformalClassifier(rf_rep, alpha=0.10, random_state=seed)
        mcp_rep.conformalize(X_c, y_c)
        y_ps_e_rep = mcp_rep.predict_set(X_te)
        res_e = evaluate_mondrian_coverage(y_te, y_ps_e_rep, classes=rf_rep.classes_)
        stats_accumulator["mondrian_cov_marg"].append(res_e["marginal_coverage"])
        stats_accumulator["mondrian_cov_norm"].append(res_e["per_class_coverage"]["Normal"])
        stats_accumulator["mondrian_cov_susp"].append(res_e["per_class_coverage"]["Suspect"])
        stats_accumulator["mondrian_cov_path"].append(res_e["per_class_coverage"]["Pathological"])
        stats_accumulator["mondrian_size"].append(res_e["avg_set_size"])

        # Methods A-E
        # Method A
        stats_accumulator["ae_cov_A"].append(1.0)
        stats_accumulator["ae_risk_A"].append(1.0 - acc_rep)

        # Method E singleton
        y_ps_e_cal_rep = mcp_rep.predict_set(X_c)
        cov_e_cal_rep = float(np.mean(np.sum(y_ps_e_cal_rep, axis=1) == 1))
        sizes_e_rep = np.sum(y_ps_e_rep, axis=1)
        mask_e_rep = (sizes_e_rep == 1)
        cov_e_rep = float(np.mean(mask_e_rep))
        risk_e_argmax = float(np.mean(preds_te[mask_e_rep] != y_te[mask_e_rep])) if np.sum(mask_e_rep) > 0 else np.nan
        preds_e_single = np.array([rf_rep.classes_[np.argmax(y_ps_e_rep[i])] if mask_e_rep[i] else -1 for i in range(len(y_te))])
        risk_e_single = float(np.mean(preds_e_single[mask_e_rep] != y_te[mask_e_rep])) if np.sum(mask_e_rep) > 0 else np.nan
        stats_accumulator["ae_cov_E"].append(cov_e_rep)
        stats_accumulator["ae_risk_E_argmax"].append(risk_e_argmax)
        stats_accumulator["ae_risk_E_single"].append(risk_e_single)

        # Abstention audit enrichment for E
        mask_abst_e = ~mask_e_rep
        n_abst = np.sum(mask_abst_e)
        if n_abst > 0 and (1.0 - acc_rep) > 0:
            err_abst = np.mean(preds_te[mask_abst_e] != y_te[mask_abst_e])
            stats_accumulator["enrichment_E"].append(err_abst / (1.0 - acc_rep))

        # Method D singleton
        sizes_d_rep = np.sum(y_ps_d_rep, axis=1)
        mask_d_rep = (sizes_d_rep == 1)
        cov_d_rep = float(np.mean(mask_d_rep))
        risk_d_argmax = float(np.mean(preds_te[mask_d_rep] != y_te[mask_d_rep])) if np.sum(mask_d_rep) > 0 else np.nan
        preds_d_single = np.array([rf_rep.classes_[np.argmax(y_ps_d_rep[i])] if mask_d_rep[i] else -1 for i in range(len(y_te))])
        risk_d_single = float(np.mean(preds_d_single[mask_d_rep] != y_te[mask_d_rep])) if np.sum(mask_d_rep) > 0 else np.nan
        stats_accumulator["ae_cov_D"].append(cov_d_rep)
        stats_accumulator["ae_risk_D_argmax"].append(risk_d_argmax)
        stats_accumulator["ae_risk_D_single"].append(risk_d_single)

        # Method B (Fixed & Cal-Matched)
        probs_c = rf_rep.predict_proba(X_c)
        conf_c = np.max(probs_c, axis=1)

        mask_b_f = (conf_te >= 0.80)
        stats_accumulator["ae_cov_B_fixed"].append(float(np.mean(mask_b_f)))
        stats_accumulator["ae_risk_B_fixed"].append(float(np.mean(preds_te[mask_b_f] != y_te[mask_b_f])) if np.sum(mask_b_f) > 0 else np.nan)

        taus_sub = np.linspace(0.40, 0.99, 100)
        t_b_m = taus_sub[np.argmin([abs(np.mean(conf_c >= t) - cov_e_cal_rep) for t in taus_sub])]
        mask_b_m = (conf_te >= t_b_m)
        stats_accumulator["ae_cov_B_match"].append(float(np.mean(mask_b_m)))
        stats_accumulator["ae_risk_B_match"].append(float(np.mean(preds_te[mask_b_m] != y_te[mask_b_m])) if np.sum(mask_b_m) > 0 else np.nan)

        # Method C (Platt)
        cal_lr_rep = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
        cal_lr_rep.fit(probs_c, y_c)
        cal_probs_c = cal_lr_rep.predict_proba(probs_c)
        cal_conf_c = np.max(cal_probs_c, axis=1)
        cal_probs_te = cal_lr_rep.predict_proba(probs_te)
        cal_conf_te = np.max(cal_probs_te, axis=1)
        cal_preds_te = cal_lr_rep.classes_[np.argmax(cal_probs_te, axis=1)]

        mask_c_f = (cal_conf_te >= 0.80)
        stats_accumulator["ae_cov_C_fixed"].append(float(np.mean(mask_c_f)))
        stats_accumulator["ae_risk_C_fixed"].append(float(np.mean(cal_preds_te[mask_c_f] != y_te[mask_c_f])) if np.sum(mask_c_f) > 0 else np.nan)

        t_c_m = taus_sub[np.argmin([abs(np.mean(cal_conf_c >= t) - cov_e_cal_rep) for t in taus_sub])]
        mask_c_m = (cal_conf_te >= t_c_m)
        stats_accumulator["ae_cov_C_match"].append(float(np.mean(mask_c_m)))
        stats_accumulator["ae_risk_C_match"].append(float(np.mean(cal_preds_te[mask_c_m] != y_te[mask_c_m])) if np.sum(mask_c_m) > 0 else np.nan)

    repeated_results = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))} for k, v in stats_accumulator.items()}
    rep_out = PROCESSED_DIR / "repeated_splits_100_results.json"
    with open(rep_out, "w") as f:
        json.dump(repeated_results, f, indent=2)
    print(f"Saved 100 repeated splits results to: {rep_out}")

    print("\n--- 100 REPEATED SPLITS SUMMARY (MEAN ± STD) ---")
    print(f"Baseline Acc: {repeated_results['baseline_acc']['mean']:.2%} ± {repeated_results['baseline_acc']['std']:.2%}")
    print(f"Baseline Recall Pathological: {repeated_results['baseline_rec_path']['mean']:.2%} ± {repeated_results['baseline_rec_path']['std']:.2%}")
    print(f"Global LAC Cov Pathological: {repeated_results['global_cov_path']['mean']:.2%} ± {repeated_results['global_cov_path']['std']:.2%}")
    print(f"Mondrian LAC Cov Pathological: {repeated_results['mondrian_cov_path']['mean']:.2%} ± {repeated_results['mondrian_cov_path']['std']:.2%}")
    print(f"Method E Coverage: {repeated_results['ae_cov_E']['mean']:.2%} ± {repeated_results['ae_cov_E']['std']:.2%}")
    print(f"Method E Risk (RF argmax): {repeated_results['ae_risk_E_argmax']['mean']:.2%} ± {repeated_results['ae_risk_E_argmax']['std']:.2%}")
    print(f"Method E Risk (Singleton): {repeated_results['ae_risk_E_single']['mean']:.2%} ± {repeated_results['ae_risk_E_single']['std']:.2%}")
    print(f"Method E Enrichment: {repeated_results['enrichment_E']['mean']:.2f}x ± {repeated_results['enrichment_E']['std']:.2f}x")

    # 10. RQ3 Sensitivity Experiment (Complete calibration partition at 100%)
    print("\n--- RUNNING RQ3 CALIBRATION-SET SIZE SENSITIVITY EXPERIMENT ---")
    df_sensitivity = run_calibration_sensitivity_experiment(
        rf, X_cal, y_cal, X_test, y_test, alpha=0.10, fractions=[0.15, 0.25, 0.35, 0.50, 0.75, 1.00], n_trials=30, random_state=RANDOM_STATE
    )
    sens_out = PROCESSED_DIR / "rq3_sensitivity_results.csv"
    df_sensitivity.to_csv(sens_out, index=False)
    print(f"Saved RQ3 sensitivity data to: {sens_out}")
    print(df_sensitivity[["cal_fraction", "n_cal_total", "n_patho_cal_approx", "patho_cov_mean", "patho_cov_std", "patho_size_mean", "patho_size_std"]].to_string(index=False))

    # 11. Regenerate All Publication Figures
    print("\n--- REGENERATING PUBLICATION FIGURES IN reports/figures/ ---")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Fig 2: Global Conformal
    fig2_path = FIGURES_DIR / "fig2_global_conformal_comparison.png"
    plot_global_conformal_comparison(df_global_lac, df_global_aps, save_path=fig2_path)
    plt.close("all")

    # Fig 3: Mondrian Comparison
    fig3_path = FIGURES_DIR / "fig3_mondrian_conformal_comparison.png"
    plot_mondrian_comparison(df_global_lac, df_mondrian_lac, save_path=fig3_path)
    plt.close("all")

    # Fig 4: Sensitivity Curves
    fig4_path = FIGURES_DIR / "fig4_rq3_calibration_size_sensitivity.png"
    plot_sensitivity_curves(df_sensitivity, save_path=fig4_path)
    plt.close("all")

    # Fig 5: Risk-Coverage Curves & Abstention
    fig5_path = FIGURES_DIR / "fig5_selective_abstention_risk_coverage.png"
    rc_curves = compute_risk_coverage_curves(rf, X_cal, y_cal, X_test, y_test)
    plot_risk_coverage_curves(rc_curves, baseline_risk=base_res["error_rate"], df_ae_comparison=df_ae_matched, save_path=fig5_path)
    plt.close("all")

    # Also Fig 5B (dedicated bar chart comparison)
    fig5b_path = FIGURES_DIR / "fig5b_abstention_method_comparison.png"
    fig5b, ax = plt.subplots(figsize=(10, 5.5))
    bar_x = np.arange(len(df_ae_matched))
    width = 0.35
    ax.bar(bar_x - width/2, df_ae_matched["Coverage"] * 100.0, width, label="Coverage (%)", color="#1f77b4", alpha=0.85)
    ax.bar(bar_x + width/2, df_ae_matched["Selective Risk (RF argmax)"] * 100.0, width, label="Selective Risk (%)", color="#d62728", alpha=0.85)
    ax.set_xticks(bar_x)
    labels = ["A: Baseline", "B: Raw Conf", "C: Calibrated", "D: Global CP", "E: Mondrian CP"]
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title(f"Methods A-E Comparison (Methods B & C cal-matched to Method E: {coverage_target_e_cal:.1%})", fontweight="bold", fontsize=12)
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", framealpha=0.9)
    for i in range(len(df_ae_matched)):
        c_val = df_ae_matched["Coverage"].iloc[i] * 100.0
        r_val = df_ae_matched["Selective Risk (RF argmax)"].iloc[i] * 100.0
        ax.text(i - width/2, c_val + 1.5, f"{c_val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, r_val + 1.5, f"{r_val:.1f}%", ha="center", fontsize=9, fontweight="bold")
    fig5b.tight_layout()
    fig5b.savefig(fig5b_path, dpi=300, bbox_inches="tight")
    plt.close(fig5b)
    print(f"Saved Fig 5B to: {fig5b_path}")

    print("\n" + "=" * 80)
    print("MASTER EVALUATION AND SYNTHESIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_evaluation()
