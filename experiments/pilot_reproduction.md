# Reproduce the small pilot

Run commands from this `project` directory. Python 3.12 and macOS `bsdtar` are required for the existing raw archive streaming code. Recorded package versions are in `requirements-pilot-lock.txt`; audit dependencies remain in `requirements-audit.txt`. The current installation keeps model dependencies in `.model_deps` and audit dependencies in `.audit_deps`. No global Python installation is modified.

```sh
export PYTHONPATH="$PWD/.model_deps:$PWD/.audit_deps"
export DYLD_LIBRARY_PATH="$PWD/.model_deps/torch/lib"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PYTHON_BIN=/Users/rajatgarg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

`DYLD_LIBRARY_PATH` lets the macOS XGBoost wheel load the OpenMP runtime bundled with the installed PyTorch wheel. For another operating system, use that platform's compatible wheels/runtime. Runtime portability is not guaranteed by a version list alone.

Existing panels can be used directly. To regenerate them from the preserved raw archives:

```sh
"$PYTHON_BIN" src/preprocessing/pilot_panels.py dunnhumby
"$PYTHON_BIN" src/preprocessing/pilot_panels.py favorita
"$PYTHON_BIN" src/preprocessing/pilot_panels.py uci
```

Favorita makes two bounded streaming passes through compressed train data. It does not materialize the full uncompressed CSV. UCI and Dunnhumby raw tables may be read before applying the chronological development cutoff, but no held-out values are used in transformations, cohort selection or panels. Dataset schema and row/date totals were already established by the completed audit.

Before **any** fitting:

```sh
"$PYTHON_BIN" src/feature_engineering/leakage_audit.py
```

This must pass. Each learner checks the audit's tracked code/data/configuration checksums before fitting; changing tracked files requires rerunning the audit. The gate covers future-value invariance, training-label chronology, encodings, transforms, selection cutoff, masks and Hybrid calibration boundaries. It applies to the pilot, not every possible future pipeline change.

Then run sequentially to limit RAM:

```sh
"$PYTHON_BIN" -u src/evaluation/run_pilot.py dunnhumby > logs/pilot/dunnhumby_run.log 2>&1
"$PYTHON_BIN" -u src/evaluation/run_pilot.py favorita > logs/pilot/favorita_run.log 2>&1
"$PYTHON_BIN" -u src/evaluation/run_pilot.py uci > logs/pilot/uci_run.log 2>&1
"$PYTHON_BIN" src/reporting/verify_saved_models.py
"$PYTHON_BIN" src/reporting/build_pilot_report.py
```

These commands overwrite canonical **pilot** artifacts with the same declared experiment. Archive an existing run first when experimenting; never overwrite a poor result to conceal it. The initial Dunnhumby attempt with failed child-process RSS monitoring is retained in `results/pilot/attempt_1_monitor_issue/`. Its memory measurements are invalid; corrected monitoring measures only the training process, where these threaded learners run. No model, cohort, seed or hyperparameter was changed to address the monitor issue.

## What is fixed

- Exactly three datasets, five individual model architectures and their five-component Hybrid.
- Cohorts selected before tuning using training eligibility, volume strata and a deterministic hash. No error-based substitutions.
- The unchanged train/tuning/calibration/final-test split manifest.
- One calibration fold for weights and the next calibration fold for pilot evaluation.
- Per-dataset targets and scoring masks shared across every model.
- Conservative historical business covariates; no future realized promotion, oil, transactions, customers or holiday values in predictors.
- One correction trial: first-calibration pooled Hybrid residual offset, then evaluation on the second fold. It is not retained for final testing.

Metrics use the shared successful/scorable six-model intersection as the primary population; own-successful coverage is also reported. Percentage metrics are undefined when their required denominators are absent. Failed component forecasts propagate to Hybrid and are never silently substituted.

## Limits before full training

The scripts intentionally implement a feasibility pilot only. They do not provide a production-scale full-experiment runner. Five neural epochs and fixed simple settings are not evidence of fully tuned model potential. Two model-fitting origins provide only one independent Hybrid evaluation origin. Full-scale preprocessing, bounded global training windows, storage budgeting, declared tuning trials, additional calibration validation and a true final freeze remain future work.

The model artifacts include fitted components, training-only transform state and calibration weights. Feature names/order are defined in `src/feature_engineering/pilot_features.py`; architectures and configurations are recorded in `models/pilot/`. The pilot snapshot records checksums and versions. SARIMA pickle and PyTorch artifacts should only be loaded from this trusted local experiment.
