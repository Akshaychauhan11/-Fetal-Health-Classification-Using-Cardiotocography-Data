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

**Owner: Member 3**

- [ ] Install and test MAPIE on the calibration set
- [ ] Implement LAC conformal prediction — global (all classes together)
- [ ] Implement APS (Adaptive Prediction Sets) — global
- [ ] Compare LAC vs APS: coverage, average prediction-set size
- [ ] Implement Mondrian (class-conditional) conformal prediction
- [ ] Compare global vs Mondrian coverage, specifically for the Pathological class
- [ ] Calibration-size sensitivity experiment: small / medium / large calibration subsets → measure coverage stability + prediction-set size variability for Pathological class
- [ ] Report results as mean ± standard deviation across repeated runs
- [ ] Commit + push

---

## Phase 4 — Selective Abstention (Objective 4, part 1)

**Owner: Member 3 + Member 2**

- [ ] Implement accept/abstain decision logic based on conformal prediction-set size / confidence
- [ ] Build the A–E method comparison (baseline / confidence-threshold / calibrated-threshold / global CP / Mondrian CP)
- [ ] Compute coverage and selective risk for each method
- [ ] Plot risk–coverage curve
- [ ] Commit + push

---

## Phase 5 — Explainability (Objective 4, part 2)

**Owner: Member 4**

- [ ] Apply SHAP to the best model
- [ ] Generate feature-importance plots for accepted predictions
- [ ] Show example: one accepted case with top contributing features explained
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
