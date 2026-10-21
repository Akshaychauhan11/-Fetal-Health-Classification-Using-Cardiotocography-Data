# RULES.md — Team & Experimental Rules

## 1. Golden Rule of This Project

**The test set is sacred. It is touched exactly once, at the very end of each objective, for final reporting — never for tuning, never for model selection, never "just to check."**
Every other rule below exists to protect this one.

---

## 2. Data & Experimental Rules

1. **Split before anything else.** Train / Calibration / Test split happens immediately after loading the cleaned dataset, using `stratify=y`, before any preprocessing, scaling, resampling, or feature selection is fit.
2. **Fit transformations on training data only.** Any scaler, encoder, or resampler (e.g. SMOTE) is `.fit()` on the training set only, then `.transform()`-ed onto calibration/test — never fit on calibration or test data. Fitting on the whole dataset before splitting is data leakage and invalidates results.
3. **The calibration set is not the test set.** It exists specifically for probability calibration and conformal-prediction calibration. Never evaluate final results on it.
4. **Report per-class metrics, always.** Overall accuracy is never reported alone — precision/recall/F1 must be broken out for Normal, Suspect, and Pathological every time.
5. **Fixed random seeds everywhere.** `random_state=42` (or a documented team-wide seed) on every split, every model, every resampling step — so results are reproducible and comparable across runs and across teammates' machines.
6. **Report mean ± standard deviation** wherever an experiment is repeated (e.g. across cross-validation folds or multiple seeds) — never a single lucky number.
7. **No invented results.** If an experiment doesn't work, or a hypothesis isn't supported, that is reported honestly — never adjusted, cherry-picked, or hidden to make the project look better.
8. **No overclaiming novelty.** Every "this hasn't been done before" claim must be traceable to an actual literature search, not assumed. When in doubt, phrase it as "to the best of our knowledge, based on a structured search."

## 3. Git / GitHub Rules

1. **Branch per task, not per person.** Branch names: `feature/<short-task-name>` (e.g. `feature/baseline-models`, `feature/calibration-check`). Avoid long-lived personal branches that drift from `main`.
2. **Commit early, commit often.** A commit should represent one logical step (e.g. "add stratified split function"), not a full day's work in one blob — this also builds your visible progress history for mentor check-ins.
3. **Pull before you push.** Always `git pull` on `main` before starting new work to avoid painful merge conflicts.
4. **Pull requests for anything touching shared code** (`src/`, `backend/`). Solo notebook experimentation can go straight to a personal branch, but anything merging into `main` gets at least one teammate's quick look.
5. **Never commit secrets or large data files.** `.gitignore` covers `venv/`, `data/processed/` (if large), `.env`, and any API keys.
6. **Commit messages describe *what* and *why***, briefly — e.g. `"Add class_weight='balanced' to RF baseline — improves Pathological recall"`, not `"update"` or `"fix stuff"`.

## 4. Code & Notebook Rules

1. **Notebooks are for exploration, `src/` is for anything reused.** Once a function works in a notebook and you'll use it again (e.g. the data split, the calibration check), move it into `src/` as a proper `.py` function and import it — don't copy-paste code across notebooks.
2. **Every notebook starts with a markdown cell** explaining: what this notebook does, which objective it belongs to, and what its outputs are.
3. **Save model artifacts, don't retrain blindly.** Once a model is trained, save it (`joblib.dump`) into `models/` so the backend and later notebooks can load it without retraining every time.
4. **No hardcoded file paths.** Use relative paths from the project root so the code runs on all 4 laptops without editing.

## 5. Team Process Rules

1. **Every teammate understands the full methodology**, even if one person "owns" a component — this project must be defensible as a team in a viva, not as four disconnected parts.
2. **Update the team before switching direction.** If literature review turns up something that changes an approach (like the class-imbalance discussion), it gets discussed and logged before anyone starts rebuilding — no silent pivots.
3. **Flag blockers immediately**, don't sit on them — a half-day lost to a stuck teammate is recoverable; a blocker discovered the night before a deadline isn't.
4. **Every claim in the written report must map to actual code/results** that exist in the repo — no describing a component in the synopsis that wasn't actually built or tested.

## 6. Team Role Ownership (primary responsibility, not exclusive)

- **Member 1** — Dataset, EDA, preprocessing, class-distribution analysis, train/calibration/test split design.
- **Member 2** — Baseline ML models, hyperparameter tuning, probability calibration, classification evaluation.
- **Member 3** — Conformal uncertainty: LAC, APS, global and Mondrian conformal prediction, abstention, coverage/selective-risk analysis, calibration-size sensitivity.
- **Member 4** — Explainability (SHAP), FastAPI backend, React frontend, SQLite integration, final prototype.
