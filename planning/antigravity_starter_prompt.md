# Antigravity Starter Prompt

Copy everything below into Antigravity to kick off the build. It references the 4 planning docs, so keep `prd.md`, `architecture.md`, `rules.md`, and `tasks.md` in your project root before running this.

---

```
I'm building a research ML project: an uncertainty-aware fetal health classifier
using Cardiotocography (CTG) data. I have four planning documents in this project
root that define everything — read them first, in this order, before writing any
code:

1. prd.md — what we're building and why
2. architecture.md — the tech stack, folder structure, data pipeline, and API design
3. rules.md — experimental rules (data leakage prevention, split discipline, etc.)
   and coding/git conventions you must follow
4. tasks.md — the full task breakdown by phase

IMPORTANT — scope for right now:
Only work on Phase 0 (Setup) and Phase 1 (Baseline Models) from tasks.md.
Do NOT touch calibration, conformal prediction, abstention, SHAP, the backend,
or the frontend yet — those come in later phases. I need Phase 1 done well and
correctly, not everything done badly.

What I need you to do, in order:

1. Create the folder structure exactly as described in architecture.md.
2. Create a venv, generate requirements.txt with the packages listed in
   architecture.md, and install them.
3. I will place the dataset file (fetal_health.csv) into data/raw/ — check that
   it loaded correctly and confirm the shape is (2126, 22): 21 features +
   1 target column called `fetal_health`.
4. Write src/data/load.py — a function to load and lightly validate the dataset
   (check for nulls, confirm expected columns).
5. Write src/data/split.py — a stratified train/calibration/test split function.
   Follow the golden rule in rules.md: the split happens before any other
   processing, and the test set is never touched again until final reporting.
   Use a fixed random_state=42 everywhere. Split ratio: 60% train / 20%
   calibration / 20% test (two-step stratified split: first carve off 40%
   as temp with test_size=0.4, then split that temp set 50/50 into
   calibration and test).
6. Create notebooks/01_eda.ipynb — load the data, check nulls, plot the class
   distribution (fetal_health value counts) as a bar chart, and clearly show
   the class imbalance numerically.
7. Create notebooks/02_baseline_models.ipynb that:
   - Uses the split from src/data/split.py
   - Trains ONLY Logistic Regression for now, with class_weight='balanced'.
     Do not train Random Forest or XGBoost — my teammates are training those
     two separately on their own machines using the same split function
     (same random_state=42, same 60/20/20 ratio), so I only need Logistic
     Regression here.
   - Prints a full per-class classification report (precision, recall, F1)
     on the test set — not just overall accuracy
   - Plots a confusion matrix for this model
   - Saves the trained model into models/ using joblib

After each step, briefly tell me what you did and ask before moving to the
next one — I'm new to ML and want to understand what's happening, not just
have code appear.

IMPORTANT — do not run any git commands (no git init, add, commit, push, or
branch creation) and do not create or edit .gitignore. I will handle all
git/version-control steps myself. Just create and edit the project files.

Do not implement anything from Phase 2 onward (calibration, conformal
prediction, Mondrian CP, abstention, SHAP, FastAPI, React) until I explicitly
ask for it.
```
