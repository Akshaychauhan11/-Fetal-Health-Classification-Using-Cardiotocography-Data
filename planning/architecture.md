# ARCHITECTURE.md — System & Project Structure

## 1. Locked Tech Stack (with reasoning)

| Layer | Choice | Why (vs. rejected alternatives) |
|---|---|---|
| IDE / Environment | **Google Antigravity** (local, VS Code-based, agentic) | Agent can help write/debug code — useful for a first-time ML team. Rejected Colab (no local debugging, awkward for the later full-stack app). |
| Env management | `venv` + `requirements.txt` | Simplest way to keep 4 laptops in sync. Rejected conda/Poetry — unnecessary complexity for a pure-Python project. |
| Language | Python 3 | Every required library is Python-first. |
| Experimentation | Jupyter notebooks (`.ipynb`) for EDA/experiments → `.py` modules for the final pipeline | Notebooks give inline visual feedback while learning; scripts needed later for clean imports into the API. |
| Data handling | pandas, numpy | Standard, no real alternative at this data size. |
| Baseline models | scikit-learn (Logistic Regression, Random Forest) + XGBoost | Consistent `.fit()`/`.predict()` API; strong tabular baselines. Rejected TensorFlow/PyTorch — dataset (~2,126 rows) is too small to justify deep learning; harder to calibrate/interpret. |
| Imbalance handling | `class_weight='balanced'` (cost-sensitive learning) | Zero new dependencies, directly addresses minority-class neglect. SMOTE/oversampling can be added later as a comparison, not required for v1. |
| Calibration | `sklearn.calibration` (Brier score, `calibration_curve`) | Already inside the library in use. |
| Conformal prediction | **MAPIE** | sklearn-compatible, supports LAC, APS, and Mondrian (class-conditional) conformal prediction out of the box. Rejected hand-written conformal prediction — too risky for a team new to ML; a bug here would invalidate the core research claim. |
| Explainability | SHAP | Theoretically grounded (Shapley values), more stable than LIME — important since this project's whole theme is *stability/trustworthiness*. |
| Visualization | matplotlib, seaborn | Standard for static report/paper figures. |
| **Backend** | **FastAPI** | Built-in validation (Pydantic) for the 21-feature input, free interactive docs at `/docs`, structured JSON output fits our prediction+confidence+decision+explanation response shape. Rejected Flask — same result, more manual boilerplate. |
| **Database** | **SQLite + SQLModel** | Needed to store abstained/flagged cases for later review — a real requirement, not decoration. SQLite = zero setup, one file, sufficient for a research prototype. Rejected Postgres/MySQL — unnecessary server setup for this scale. SQLModel reuses the same Pydantic-style syntax as FastAPI, so no second ORM language to learn. |
| **Frontend** | **React** (plain, no Redux/router needed) | Only two real screens (input form, result panel) — no need for heavy tooling. |
| Version control | Git + GitHub, feature-branch workflow | Team is already comfortable; branches prevent 4 people overwriting each other's work. |

**Explicitly not used**: Streamlit (skipped entirely — goes straight to FastAPI + React once the prototype phase starts, no throwaway intermediate version).

---

## 2. Folder Structure

```
fetal-health-uncertainty/
│
├── data/
│   ├── raw/                     # original fetal_health.csv (never edited)
│   └── processed/                # cleaned / split data (train/cal/test), git-ignored if large
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_random_forest.ipynb
│   ├── 04_xgboost.ipynb
│   ├── 05_calibration.ipynb
│   ├── 06_conformal_prediction.ipynb
│   ├── 07_mondrian_conformal.ipynb
│   ├── 08_calibration_size_sensitivity.ipynb
│   ├── 09_abstention.ipynb
│   └── 10_shap_explainability.ipynb
│
├── src/
│   ├── data/
│   │   ├── load.py               # load + clean CTG data
│   │   └── split.py              # stratified train/calibration/test split
│   ├── models/
│   │   ├── train_baseline.py     # train LogReg / RF / XGBoost
│   │   └── registry.py           # save/load trained model artifacts
│   ├── calibration/
│   │   └── calibrate.py          # Brier score, ECE, reliability diagram
│   ├── conformal/
│   │   ├── global_cp.py          # LAC / APS via MAPIE
│   │   └── mondrian_cp.py        # class-conditional conformal prediction
│   ├── abstention/
│   │   └── selective.py          # accept/abstain decision logic
│   └── explainability/
│       └── shap_explain.py
│
├── models/                       # saved trained model artifacts (.pkl / .joblib)
│
├── backend/                      # FastAPI app
│   ├── main.py
│   ├── schemas.py                # Pydantic request/response models
│   ├── db.py                     # SQLModel + SQLite setup
│   └── predict.py                # loads pipeline, runs prediction end-to-end
│
├── frontend/                     # React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── InputForm.jsx
│   │   │   └── ResultPanel.jsx
│   │   └── App.jsx
│   └── package.json
│
├── reports/
│   ├── synopsis/                  # research synopsis docs
│   └── figures/                   # exported plots for the report
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. Data / Model Pipeline (conceptual flow)

```
CTG Dataset (fetal_health.csv)
        ↓
Preprocessing & EDA  (src/data/load.py)
        ↓
Stratified split: Train / Calibration / Test  (src/data/split.py)
        ↓
Baseline ML Models: LogReg, RF, XGBoost  (src/models/train_baseline.py)
        ↓
Probability Calibration  (src/calibration/calibrate.py)
        ↓
Conformal Prediction — LAC / APS  (src/conformal/global_cp.py)
        ↓
Class-Conditional (Mondrian) Analysis  (src/conformal/mondrian_cp.py)
        ↓
Selective Prediction: Accept / Abstain  (src/abstention/selective.py)
        ↓
Explainability (SHAP)  (src/explainability/shap_explain.py)
        ↓
FastAPI backend serves this pipeline → React frontend displays result → SQLite logs abstained cases
```

## 4. API Contract (FastAPI)

```
POST /predict

Request body (21 CTG features):
{
  "baseline_value": 120,
  "accelerations": 0.003,
  "fetal_movement": 0.000,
  "uterine_contractions": 0.005,
  "light_decelerations": 0.000,
  "severe_decelerations": 0.000,
  "prolongued_decelerations": 0.000,
  "abnormal_short_term_variability": 73,
  "mean_value_of_short_term_variability": 0.5,
  "percentage_of_time_with_abnormal_long_term_variability": 43,
  "mean_value_of_long_term_variability": 2.4,
  "histogram_width": 64,
  "histogram_min": 62,
  "histogram_max": 126,
  "histogram_number_of_peaks": 2,
  "histogram_number_of_zeroes": 0,
  "histogram_mode": 120,
  "histogram_mean": 137,
  "histogram_median": 121,
  "histogram_variance": 73,
  "histogram_tendency": 1
}

Response:
{
  "prediction_set": ["Suspect", "Pathological"],
  "confidence": "low",
  "decision": "ABSTAIN",
  "top_features": ["abnormal_short_term_variability", "prolongued_decelerations"]
}
```

## 5. Database Schema (SQLite, via SQLModel)

```
Table: predictions
- id: int (primary key)
- timestamp: datetime
- input_features: JSON (the 21 submitted values)
- prediction_set: JSON (list of classes in the conformal set)
- confidence_level: str
- decision: str ("ACCEPT" | "ABSTAIN")
- top_features: JSON (SHAP top contributing features)
- reviewed: bool (default False — set True once a human reviews an abstained case)
```

## 6. Environment Setup (once, per laptop)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (initial):
```
pandas
numpy
scikit-learn
xgboost
mapie
shap
matplotlib
seaborn
fastapi
uvicorn
sqlmodel
```
