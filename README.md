# Wetland Detection with Classical ML

**A scikit-learn lab notebook that classifies land samples as wetland or dry land from four spectral bands and compares a decision tree, logistic regression, an SVM, bagging and a random forest on the same test set.**

## Overview

The task is binary wetland mapping: every row of the dataset is one land sample with four numeric
features (near-infrared, red, green and blue band values) and a label, `0` for dry land and `1` for
wetland. The notebook `Lab1-Final.ipynb` works through four parts: train a decision tree and compute
accuracy / precision / recall / F1 for the wetland class; reproduce overfitting by retraining on a
much smaller training file and then tame it with `min_samples_split=30`; compare logistic regression
and an RBF SVM against the tree; and finally run two ensembles (bagging of decision trees and a random
forest). Everything is plain pandas + scikit-learn in a Python 3.8 kernel. The notebook keeps the
original lab scaffolding ("COMPLETE CODES BELOW" prompts and a short written answer on when accuracy
is a misleading metric), so it reads as a completed course lab rather than a standalone library.

## What it does

- Loads `train.csv`, `test.csv` and `smalltrain.csv` (header-less, five columns) with pandas and
  splits them into `X_train`/`Y_train` and `X_test`/`Y_test`.
- Part 1: `DecisionTreeClassifier()` with default settings, scored with `metrics.accuracy_score`,
  `precision_score`, `recall_score` and `f1_score` (positive class = wetland).
- Part 2: retrains the same tree on `smalltrain.csv`, prints training-set and test-set metrics to show
  the overfitting gap, then fits `DecisionTreeClassifier(min_samples_split=30)` as the simpler model.
- Part 3: `LogisticRegression(solver='liblinear')` and `SVC(gamma='scale', C=100)` on the full
  training set.
- Part 4: `BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=1.0,
  bootstrap=True, random_state=1)` and `RandomForestClassifier(n_estimators=50)`.

## How it works

| File | Purpose |
| --- | --- |
| `Lab1-Final.ipynb` | The whole pipeline: data loading, five classifiers, metrics, written answer on metric choice |
| `train.csv`, `test.csv`, `smalltrain.csv` | Expected next to the notebook; **not included in this repository** |

Pipeline: CSV -> `DataFrame` with columns `near infra red`, `red`, `green`, `blue`, `class` ->
feature/label split -> fit model -> predict on the test file -> four scikit-learn metrics printed.
No scaling or feature engineering is applied; the four raw band values are fed straight to each model.

### Results reported in the notebook (test set, wetland = positive class)

| Model | Training file | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Decision tree (default) | `train.csv` | 0.778 | 0.734 | 0.768 | 0.751 |
| Decision tree (default) | `smalltrain.csv` | 0.701 | 0.655 | 0.660 | 0.658 |
| Decision tree, `min_samples_split=30` | `smalltrain.csv` | 0.757 | 0.688 | 0.807 | 0.743 |
| Logistic regression (`liblinear`) | `train.csv` | 0.757 | 0.731 | 0.699 | 0.714 |
| SVC (`gamma='scale'`, `C=100`) | `train.csv` | 0.810 | 0.741 | 0.867 | 0.799 |
| Bagging, 100 decision trees | `train.csv` | 0.813 | 0.778 | 0.798 | 0.788 |
| Random forest, 50 trees | `train.csv` | 0.807 | 0.771 | 0.791 | 0.781 |

The overfitting experiment also prints the tree's metrics on its own small training set: all four
are exactly 1.0, against 0.701 accuracy on the test set. Restricting splits to nodes with at least
30 samples recovers most of that gap (0.757). The ensembles and the SVM are the strongest models in
the notebook; the SVM has the highest wetland recall (0.867), bagging the highest precision (0.778).

## Running it

The three CSV files are not committed, so you need them beside the notebook first. Then:

```bash
pip install pandas scikit-learn jupyter
jupyter notebook Lab1-Final.ipynb
```

or, without touching the notebook in place:

```bash
uv run --with pandas,scikit-learn,nbconvert,ipykernel \
  jupyter nbconvert --to notebook --execute Lab1-Final.ipynb --output executed.ipynb
```

Not verified on this machine: the data files are absent, so the notebook was only read, not re-run.

## Limitations / Notes

- The dataset is not in the repository and the notebook does not say where it came from, so the
  numbers above cannot be reproduced from this repo alone.
- The decision tree, random forest and bagging base trees are unseeded (only the bagging wrapper sets
  `random_state=1`), so re-running will give slightly different metrics than the ones reported.
- The first feature's column name carries a trailing space (`'near infra red '`), copied from the lab
  template; it is harmless but easy to trip over when selecting columns.
- It is a single lab notebook, not a package: there are no tests, no CLI and no saved models.
