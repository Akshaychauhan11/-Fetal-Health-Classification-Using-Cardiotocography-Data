# PRD.md — Project Requirement Document

## Project Title
**Uncertainty-Aware and Explainable Machine Learning Framework for Fetal Health Classification Using Cardiotocography Data**

## Team
- Shambhavi Chauhan – 12515800069
- Akshay Chauhan – 12515800008
- Devansh Parmar – 12515800021
- Kavya Agarwal – 12515800034

Guides: Dr. Ruby Panwar (Program Coordinator, B.Tech CSE Hons.), Dr. Prasenjit Dhar (Class Advisor)
Institution: GLA University, Mathura

---

## 1. Problem Statement

Cardiotocography (CTG) is used to classify fetal status into **Normal / Suspect / Pathological**. Existing ML classifiers already achieve high accuracy on public CTG benchmarks, but they:
- Are forced to output one label for every case, even ambiguous ones.
- Don't report whether their confidence is statistically calibrated (trustworthy).
- Treat all classes equally, even though Pathological — the rarest and most clinically important class — is where errors matter most.

This project does **not** try to build a clinical diagnostic tool. It builds a **research prototype that knows when it doesn't know** — a fetal-health classifier that abstains (flags for human review) instead of guessing on uncertain cases, with special attention to the minority Pathological class.

## 2. Dataset

- **Source**: UCI Cardiotocography dataset (Kaggle mirror: `andrewmvd/fetal-health-classification`)
- **Size**: 2,126 records, 21 numeric CTG-derived features + 1 target column (`fetal_health`)
- **Target**: 1 = Normal, 2 = Suspect, 3 = Pathological
- **Known issue**: severe class imbalance (Normal dominates; Pathological is the smallest class)
- No missing values; dataset arrives pre-cleaned.

## 3. Objectives (final, consolidated — 4, not 8)

1. **Build and evaluate baseline ML classifiers** for CTG-based fetal-health classification (Logistic Regression, Random Forest, XGBoost), with per-class (not just overall) evaluation.
2. **Investigate whether the models' predicted confidence is statistically calibrated** (Brier score, Expected Calibration Error, reliability diagrams).
3. **Apply class-specific (Mondrian) conformal prediction** to produce statistically grounded uncertainty estimates, especially for the minority Pathological class, including studying how the amount of calibration data affects reliability.
4. **Design a selective-abstention mechanism** (accept vs. abstain) with SHAP-based explainability for accepted predictions, and package the full pipeline into a working prototype (FastAPI + React + SQLite).

## 4. Research Questions

- **RQ1**: Can an uncertainty-aware, class-specific prediction system make CTG-based fetal-health classification more reliable by identifying cases where the model should not make an automatic decision?
- **RQ2**: Does class-conditional (Mondrian) conformal prediction improve reliability for the minority classes specifically, without rejecting an excessive proportion of cases?
- **RQ3**: How does the amount of class-specific calibration data affect the stability of conformal coverage and prediction-set size, particularly for the Pathological class?

## 5. Success Criteria (what "working" means at each stage)

| Objective | Done when... |
|---|---|
| 1 | 3 baseline models trained, per-class precision/recall/F1 reported, confusion matrix produced |
| 2 | Brier score + ECE computed for best baseline; reliability diagram plotted |
| 3 | LAC + APS conformal prediction implemented; Mondrian version compared to global; calibration-set-size sensitivity tested |
| 4 | Abstain/accept logic working end-to-end; SHAP explanations shown for accepted cases; FastAPI + React + SQLite prototype functional |

## 6. Non-Goals (explicitly out of scope)

- This is **not** a clinical diagnostic tool and must never be presented as one.
- No claims of state-of-the-art accuracy — accuracy is not the research contribution.
- No maternal-health prediction (fetal health only; maternal-health integration is future work only).
- No invention of new conformal-prediction math — we apply established techniques (Vovk et al. 2005; Sadinle et al. 2019; Romano et al. 2020) to a new setting (CTG), we do not claim to invent them.
- No deep learning models unless baseline results clearly justify the added complexity (small dataset, tabular data — tree-based models are expected to be competitive).

## 7. Milestones / Timeline

| Milestone | Target | Scope |
|---|---|---|
| **Progress Check 1** | 24 Sept 2026 (~40%) | Objective 1 (baselines) + start of Objective 2 (calibration) |
| Progress Check 2 | TBD | Objective 3 (conformal prediction, Mondrian, calibration-size sensitivity) |
| Progress Check 3 | TBD | Objective 4 (abstention + SHAP) |
| Final | TBD | Full prototype (FastAPI + React + SQLite) + final research report |

## 8. Deliverables

- Trained models + evaluation notebooks (per objective)
- Final synopsis / research report (already drafted separately)
- Working prototype: FastAPI backend + React frontend + SQLite database (flagged-case log)
- GitHub repository with full commit history as evidence of progress
