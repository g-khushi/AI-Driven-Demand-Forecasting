# Latest results: frozen final-test comparison

The [final comparison report](results/final_experiment/final_comparison_report.md) and [verdict](results/final_experiment/verdict.json) complete the forecasting-only comparison on the fixed 36-series cohort across four original final-test origins per dataset. Corrected Hybrid has 2.54% lower equal-dataset normalized MAE than the corrected compact forecasting-only comparator. Native MAE is worse on Dunnhumby and Favorita and better on UCI; this is not uniform superiority.

Final-test outcomes have now been evaluated and must not be reused for tuning. The final model configurations, predictions, replay checks, correction comparisons and original split boundaries are preserved. [Reproduction and integrity details](experiments/final_experiment/README.md).

Full-population training remains incomplete because the current feature materialization exceeds available memory. Full MARIOD remains unreproduced because inventory/sensor/supply-chain inputs are missing. These limitations are explicit; the compact forecasting comparator must not be relabelled as full MARIOD.

Earlier records below describe their status at the time and are superseded by the final report.

## Previous continuation: combined-dataset Hybrid

The [combined Hybrid development report](results/combined_v2/combined_hybrid_report.md) covers shared training across the three existing 12-series cohorts, a paper-inspired Transformer component, bounded weighting/correction selection, and later calibration evaluation. This is 36 series, not the full raw population. The full MARIOD comparison is pending missing inputs or the user's choice of a clearly labelled forecasting-only reconstruction. See [reproduction assessment](experiments/combined_v2/paper_reproduction_assessment.md) and [run instructions](experiments/combined_v2/README.md).

Earlier work is preserved below.

# Demand forecasting research — validation pilot

Read the measured [pilot research report](results/pilot_research_report.md), [accuracy table](results/model_accuracy_summary.csv) and [failure coverage](results/model_failure_coverage.csv).

The completed [dataset audit](results/dataset_audit_and_experimental_roadmap.md) is preserved as historical context. This continuation builds a new, explicitly described Hybrid from Seasonal Naive, SARIMA, XGBoost, LSTM and Transformer. Inverse first-calibration-fold MAE weights are frozen before evaluation on the next fold.

The pilot uses 12 training-selected entities per dataset. It is not a full-population or final-test experiment. Full training and final-test evaluation remain disabled. No claim of Hybrid superiority, statistical significance or architectural novelty is implied.

Raw data and the original split manifest are preserved. Favorita train is streamed from `train.csv.7z`; do not extract its 5 GB CSV on the current disk.

Reproduction commands and methodological boundaries are in [pilot_reproduction.md](experiments/pilot_reproduction.md). Derived panels, models, predictions, runtime measurements, failure logs, error diagnostics, ablations and one declared correction trial are saved in their respective project directories.

The original audit scripts remain available under `src/data_loader/`; rerunning acquisition or the full audit is unnecessary for this continuation.
