# Frozen final experiment

This is the final-test continuation of the combined 36-series experiment, not a full-population run. Twelve initial-training-selected series from each dataset remain fixed. The forecasting comparator is the compact cross-modal Transformer component, transparently labelled a forecasting-only MARIOD reconstruction. It is not full MARIOD and has no simulated or fabricated sensor/inventory inputs.

## Freeze and chronology

`configuration_freeze.json` was saved before test-panel construction. It fixes the existing split manifest, Hybrid weights, correction factors and offsets, primary metric, and component hyperparameters. Comparator correction was selected using calibration folds 1–2 with the same factor grid as Hybrid (0, 0.5, 1). Calibration folds 3–4 were not used to revise the Hybrid.

The final test remains:

- Dunnhumby DAY 656–711: four 14-day origins.
- Favorita 2017-06-13–2017-08-15: four 16-day origins.
- UCI 2011-10-14–2011-12-08: four 14-day origins, with non-recording dates masked.

All five components are refitted once through the first final-test origin. Learned parameters and training-only transformations remain fixed thereafter. Later origins can observe earlier test outcomes as historical context. SARIMA refreshes its filtering state with `refit=False` and checks unchanged coefficients. No optimizer runs on test labels.

## Execution and artifacts

Use the existing bundled Python 3.12 runtime with:

```sh
export PYTHONPATH="$PWD/.model_deps:$PWD/.audit_deps"
export DYLD_LIBRARY_PATH="$PWD/.model_deps/torch/lib"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PYTHON_BIN=/Users/rajatgarg/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

From `project/`, the executed pipeline is:

```sh
"$PYTHON_BIN" src/final_experiment/prepare.py dunnhumby
"$PYTHON_BIN" src/final_experiment/prepare.py favorita
"$PYTHON_BIN" src/final_experiment/prepare.py uci
"$PYTHON_BIN" src/final_experiment/audit.py
"$PYTHON_BIN" src/final_experiment/train_once.py
"$PYTHON_BIN" src/final_experiment/predict.py
"$PYTHON_BIN" src/reporting/evaluate_final_experiment.py
```

Preparation asserts that the historical panel and cohort remain unchanged. The audit tests future-target and covariate invariance, every training-label cutoff, feature timestamps, namespaces and frozen source hashes. The forecaster replays the first origin from saved artifacts and verifies unchanged model hashes across all four origins.

`training_complete.json`, `forecast_complete.json` and `test_scored.json` prevent accidentally overwriting a completed final experiment. Do not remove these guards to tune against this test. A reproducibility rerun in a separately preserved copy is not a new untouched test; any model changes after seeing this test need a new evaluation design.

Results: `results/final_experiment/final_comparison_report.md`, `final_accuracy.csv`, `accuracy_and_correction_comparisons.csv`, `combined_accuracy.csv`, `error_analysis.csv`, `comparison_consistency.csv`, `failure_coverage.csv` and `verdict.json`. Models are under `models/final_experiment/`. The original pilot and combined development results are preserved; pre-comparator development reports are archived under `results/archive/combined_v2_before_comparator/`.

## Unfinished full-scale work

The training-only eligible population is much larger than 36 series. `full_population_feasibility.json` estimates approximately 173.4 GiB for the current implementation's feature arrays alone, versus 8 GiB available RAM. This is an implementation-specific resource estimate, not a theoretical minimum: an out-of-core implementation can reduce RAM but still requires substantial compute/storage and validation. Full-population training was not completed or disguised as this sampled run.

Full MARIOD additionally requires the missing inventory, sensor, supply-chain graph and cost inputs described in the paper assessment. A successful forecasting-only comparison cannot establish superiority over that complete system.
