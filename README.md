# Fetal Health Uncertainty Quantification & Conformal Abstention

[![Status](https://img.shields.io/badge/Status-Work%20in%20Progress%20(v0--draft)-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> [!CAUTION]
> **WORK IN PROGRESS — NUMBERS NOT FINAL**  
> This repository contains ongoing research on distribution-free uncertainty quantification, class-conditional (Mondrian) conformal prediction, and selective abstention on cardiotocography data. Numbers in preliminary drafts are undergoing final verification and peer audit.

---

## 1. Project Overview

Standard machine learning models in obstetrical care produce forced point diagnoses on cardiotocography (CTG) recordings, leading to hazardous false reassurances or false alarms. This study implements:
1. **Global vs. Mondrian Conformal Prediction:** Demonstrating the minority coverage gap of standard marginal conformal prediction and its resolution via exact finite-sample class-conditional calibration.
2. **Selective Abstention:** Benchmarking five decision policies (Methods A–E) at matched coverage with clinical error tracking (false reassurances and false alarms).
3. **Finite-Sample Sensitivity (RQ3):** Monte Carlo subsampling and bootstrap estimation of calibration set size limits in rare disease subsets.

---

## 2. Dataset

The benchmark [UCI Cardiotocography Dataset](https://archive.ics.uci.edu/dataset/193/cardiotocography) comprises $N = 2,126$ fetal heart rate recordings with 21 continuous and morphometric features and 3-class expert consensus labels:
- **Normal (1.0):** $N = 1,655$ ($77.85\%$)
- **Suspect (2.0):** $N = 295$ ($13.88\%$)
- **Pathological (3.0):** $N = 176$ ($8.28\%$)

The raw dataset is included in `data/raw/fetal_health.csv`.

---

## 3. Reproduction & Execution Protocol

Every table, array, and figure in this project is regenerated deterministically from a single master script.

### Environment Setup
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Single Master Reproduction Command
```bash
python src/evaluate_all.py
```

### Fixed Experimental Parameters
- **Global Random Seed:** `42`
- **Data Splitting Protocol:** Stratified 60% Train ($N=1,275$), 20% Calibration ($N=425$), 20% Sacred Test Split ($N=426$)
- **Base Estimator:** `RandomForestClassifier(n_estimators=300, random_state=42)`
- **Calibrator:** Multinomial Logistic Regression (`solver='lbfgs'`, fitted on calibration set probabilities)
- **Nominal Significance Level:** $\alpha = 0.10$ ($90.0\%$ target coverage guarantee)
- **Repeated Splits Benchmark:** 100 repeated stratified 60/20/20 splits (`Mean ± Std`)
- **Last Authoritative Run Date:** November 27, 2026

---

## 4. Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── fetal_health.csv              # Raw UCI CTG benchmark
│   └── processed/                        # Exported prediction arrays and repeated split records
├── drafts/
│   ├── manuscript_draft.md               # Working draft of the research paper (placeholders for unfinalized numbers)
│   └── research_findings_report.md       # Detailed technical synopsis and experiment logs
├── models/                               # Saved model checkpoints
├── notebooks/                            # Step-by-step exploratory and analysis notebooks (01 to 10)
├── reports/
│   └── figures/                          # 300 DPI publication figures (fig1 through fig5b, SHAP)
├── src/
│   ├── abstention/                       # Selective abstention policies & risk-coverage metrics
│   ├── conformal/                        # Global and Mondrian conformal prediction modules
│   ├── explainability/                   # TreeSHAP feature attribution routines
│   └── evaluate_all.py                   # Master evaluation script (single source of truth)
├── README.md
└── requirements.txt
```

---

## 5. Contact & Citation

This repository is maintained as a private research workspace prior to formal manuscript submission.
