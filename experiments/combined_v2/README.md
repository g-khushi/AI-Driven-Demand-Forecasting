# Combined Hybrid experiment

This directory records a new, bounded development experiment. The original pilot, raw datasets and final-test split manifest remain unchanged. The cohort is the existing 12 training-selected entities per dataset, 36 total. Combining three sources does not mean all raw records have been included in the forecasting cohort.

The combined panel is `data/processed/combined_v2/combined_panel.csv.gz`. Entity IDs are namespaced by source, time axes remain independent, native demand is preserved, and scaled demand uses training-only entity means. No inventories, sensor streams or real calendar dates for Dunnhumby's relative days are invented.

From the `project` directory, use the existing Python 3.12 runtime and local dependencies:

```sh
export PYTHONPATH="$PWD/.model_deps:$PWD/.audit_deps"
export DYLD_LIBRARY_PATH="$PWD/.model_deps/torch/lib"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PYTHON_BIN=/Users/rajatgarg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

Prepare (skip when verified panels already exist):

```sh
"$PYTHON_BIN" src/combined_v2/prepare_panel.py dunnhumby
"$PYTHON_BIN" src/combined_v2/prepare_panel.py favorita
"$PYTHON_BIN" src/combined_v2/prepare_panel.py uci
"$PYTHON_BIN" src/combined_v2/prepare_panel.py combine
"$PYTHON_BIN" src/combined_v2/leakage_audit.py
```

Train sequentially, preserving chronological folds:

```sh
"$PYTHON_BIN" src/combined_v2/run_components.py 1
"$PYTHON_BIN" src/combined_v2/run_components.py 2
"$PYTHON_BIN" src/reporting/freeze_combined_selection.py
"$PYTHON_BIN" src/combined_v2/run_components.py 3
"$PYTHON_BIN" src/combined_v2/run_components.py 4
"$PYTHON_BIN" src/reporting/evaluate_combined_v2.py
"$PYTHON_BIN" src/reporting/verify_combined_models.py
```

The evaluator selects weights/correction using folds 1–2 only; folds 3–4 are later development evaluation. They must not become additional tuning folds after results are seen. The original final test remains unavailable to these scripts.

The cross-modal Transformer is a compact paper-inspired component of Hybrid. It is **not full MARIOD**. A forecasting-only comparison requires the user's explicit scope choice, recorded in `comparator_authorization.json`. Do not set that record merely to make a comparison table appear. Full MARIOD needs the missing inputs described in `paper_reproduction_assessment.md`.

Rerunning overwrites canonical v2 development artifacts; archive before changing configurations. Never silently overwrite unfavorable trials. Model weights and correction choices, all candidate scores, later predictions, failure coverage and own-process RAM/runtime measurements are saved separately. The previous pilot remains a historical result and is not directly comparable because model capacity, pooling and dates differ.
