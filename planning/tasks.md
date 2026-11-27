# TASKS.md — Project Task Breakdown

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Phase 0 — Setup (do first, before any modeling)

- [ ] Create GitHub repo, add all 4 teammates as collaborators
- [x] Set up folder structure per `architecture.md`
- [x] Create `venv`, install `requirements.txt`, confirm environment runs
- [x] Download dataset (`fetal_health.csv`) into `data/raw/`
- [x] Confirm dataset shape: 2,126 rows × 21 features + 1 target (`fetal_health`)
- [x] Add `prd.md`, `architecture.md`, `rules.md`, `tasks.md` to repo root

---

## Phase 1 — Baseline Models (Objective 1) — 🎯 Target: 40% checkpoint, 24 Sept 2026

**Owner: Member 1 (data) + Member 2 (models)**

- [x] Load dataset, check for nulls, confirm clean (`notebooks/01_eda.ipynb`)
- [x] Plot class distribution (`fetal_health` value counts) — confirm and visualize imbalance
- [x] Implement stratified train/calibration/test split (`src/data/split.py`)
- [x] Train Logistic Regression (`class_weight='balanced'`)
- [x] Train Random Forest (`class_weight='balanced'`) (`notebooks/03_random_forest.ipynb`)
- [x] Train XGBoost (`notebooks/04_xgboost.ipynb`)
- [x] Generate per-class classification report (precision/recall/F1) for model
- [x] Plot confusion matrix for Logistic Regression model
- [x] Save trained model to `models/logistic_regression.joblib`
- [x] Write summary of findings for Phase 1
- [ ] Commit + push everything, tag as `checkpoint-1`

**Deliverable for mentor (24 Sept):** input → model working → raw probability output → formatted prediction, using one real test-set sample, plus the per-class results table and confusion matrix.

---

## Phase 2 — Calibration Check (Objective 2, start)

**Owner: Member 2 + Member 3**

- [x] Compute Brier score for best baseline model
- [x] Compute Expected Calibration Error (ECE)
- [x] Plot reliability diagram (`sklearn.calibration.calibration_curve`)
- [x] Document: is raw confidence actually trustworthy, or overconfident?
- [x] Commit + push

---

## Phase 3 — Conformal Prediction (Objective 3)

### Phase 3A — Global Conformal Prediction & Scoring Mechanics
**Owner: Member 3A**
- [x] Install and verify MAPIE on the calibration split (`notebooks/06_conformal_prediction.ipynb`)
- [x] Implement LAC (Least Ambiguous Classifier) non-conformity score
- [x] Implement APS (Adaptive Prediction Sets) non-conformity score
- [x] Sweep significance levels $\alpha \in [0.01, 0.25]$: plot marginal coverage vs. average set size
- [x] Evaluate set composition distributions (empty sets, singletons, multi-class sets)
- [x] Evaluate per-class coverage under global CP and document minority (Pathological) under-coverage
- [x] Refactor reusable logic into `src/conformal/global_cp.py`
- [x] Export summary curves to `reports/figures/global_conformal_comparison.png`
- [ ] PR & merge branch `feature/global-conformal`

### Phase 3B — Mondrian (Class-Conditional) CP & Sensitivity Analysis
**Owner: Member 3B**
- [x] Implement Mondrian (class-conditional) conformal prediction (`notebooks/07_mondrian_conformal.ipynb`)
- [x] Compare per-class coverage: Global CP vs. Mondrian CP (focus on Pathological class recovery)
- [x] Quantify the efficiency cost (increase in average set size) of Mondrian guarantees
- [x] Implement calibration-size sensitivity experiment (`notebooks/08_calibration_size_sensitivity.ipynb`)
- [x] Run 30–50 Monte-Carlo repetitions across small/medium/full calibration subsets
- [x] Report sensitivity metrics as mean ± standard deviation across seeds
- [x] Refactor reusable logic into `src/conformal/mondrian_cp.py`
- [x] Export Mondrian comparison and sensitivity plots to `reports/figures/mondrian_comparison.png` and `reports/figures/calibration_sensitivity.png`
- [ ] PR & merge branch `feature/mondrian-conformal`

### Phase 3 Joint Milestone
**Owners: Member 3A + Member 3B**
- [x] Produce unified Global vs. Mondrian comparison table for report/synopsis
- [x] Verify standard interface format of prediction sets for Phase 4 (Selective Abstention)
- [ ] Commit + push

---

## Phase 4 — Selective Abstention (Objective 4, part 1)

**Owner: Member 3 + Member 2**

- [x] Implement accept/abstain decision logic based on conformal prediction-set size / confidence (`src/abstention/selective.py`)
- [x] Build the A–E method comparison (baseline / confidence-threshold / calibrated-threshold / global CP / Mondrian CP)
- [x] Compute coverage and selective risk for each method (`notebooks/09_abstention.ipynb`)
- [x] Plot risk–coverage curve (`reports/figures/risk_coverage_curve.png`)
- [ ] Commit + push

---

## Phase 5 — Explainability (Objective 4, part 2)

**Owner: Member 4**

- [x] Apply SHAP to the best model (`src/explainability/shap_explain.py`)
- [x] Generate feature-importance plots for accepted predictions (`reports/figures/shap_global_importance.png`)
- [x] Show example: one accepted case with top contributing features explained (`notebooks/10_shap_explainability.ipynb`, `reports/figures/shap_local_example.png`)
- [ ] Commit + push

---

## Phase 6 — Prototype: Backend (Objective 4, part 3)

**Owner: Member 4**

- [ ] Set up FastAPI project (`backend/`)
- [ ] Define Pydantic input schema (21 CTG features) — `backend/schemas.py`
- [ ] Load saved model + calibration + conformal + SHAP pipeline on startup
- [ ] Implement `POST /predict` endpoint — full pipeline, per API contract in `architecture.md`
- [ ] Set up SQLite + SQLModel — `predictions` table
- [ ] Log every abstained case to the database
- [ ] Enable CORS for the React frontend
- [ ] Test endpoint via FastAPI's `/docs` interactive page
- [ ] Commit + push

---

## Phase 7 — Prototype: Frontend

**Owner: Member 4 (+ whoever is free)**

- [ ] Set up React app (`frontend/`)
- [ ] Build `InputForm.jsx` — form for 21 CTG features (or preset example patients)
- [ ] Build `ResultPanel.jsx` — displays prediction set, confidence, accept/abstain decision, top features
- [ ] Connect frontend to FastAPI `/predict` via `fetch`/`axios`
- [ ] Test full flow: enter data → see result → check it's logged in SQLite if abstained
- [ ] Commit + push

---

## Phase 8 — Final Report & Polish

**Owner: All**

- [ ] Update synopsis/report with real results (replace "expected outcomes" language with actual findings)
- [ ] Update literature review with any new papers found during the project
- [ ] Prepare final presentation
- [ ] Final repo cleanup + README
