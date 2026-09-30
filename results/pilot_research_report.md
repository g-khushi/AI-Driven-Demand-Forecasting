# Demand forecasting: measured feasibility pilot

**Status:** preprocessing, leakage gates, two chronological calibration folds and all six model pipelines have been exercised on 12 training-selected entities per dataset. These are preliminary validation results, not final-test results or full-population research conclusions.

## Hybrid architecture and authorization

The repository search found no executable Hybrid implementation. The user clarified that a new Hybrid must be built. The starting architecture was stated before implementation: a nonnegative weighted combination of Naive, SARIMA, XGBoost, LSTM and Transformer. Each weight is proportional to the inverse pooled MAE on the first calibration fold, normalized to sum to one. Weights are frozen before the second calibration fold. Near-zero calibration errors share all weight; epsilon is recorded. This is a transparent starting forecast combination, not a claim of architectural novelty.

Each component is refitted through the second origin. Weights remain those learned from the earlier out-of-sample predictions. If any component fails, the full Hybrid cell is missing; no silent replacement or weight renormalization is used.

## Design and scope

The original split manifest is preserved. Tuning periods are part of historical training at these calibration origins, but no hyperparameter search was performed. Fixed small configurations use seed 20260922, CPU with two threads, five neural epochs, a 28-day sequence, training windows every seven days, and no random shuffle. The pilot evaluates Dunnhumby days 614–627, Favorita 2017-04-26–2017-05-11 and UCI 2011-09-02–2011-09-15. The preceding calibration fold is used only for Hybrid calibration and the declared correction trial. Final-test periods were not transformed into model panels or forecast.

Cohorts were fixed using only the initial training period: four eligible entities from each volume tercile, ordered by a deterministic hash. Selection did not use forecasting errors. Panels represent pooled household product purchases for Dunnhumby, store-item sales for Favorita, and product purchases for UCI. They are not directly comparable physical units across datasets.

Missing entity-days become conditional sales zeros only after first positive history and only on globally recorded days. Global gaps are masked. This complete-capture assumption does not identify stock availability or latent demand; availability remains unknown for unrecorded entity-days. UCI non-recording days are excluded from scoring, so its pilot describes recorded trading days. Returns and excluded records have separate ledgers; no raw files were edited. Favorita fractional units and perishable weights are preserved.

## Table 1 — Dataset and pilot characteristics

| dataset | raw_original_rows_from_completed_audit | development_input_rows | eligible_entity_count | pilot_entity_count | final_model_ready_rows | observed_panel_rows | assumed_zero_rows | masked_global_gap_rows | panel_t_min | panel_t_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | 2595732 | 2246408 | 3778 | 12 | 7122 | 2116 | 5006 | 0 | 1 | 627 |
| favorita | 125497040 | 115351331 | 147289 | 12 | 11675 | 8733 | 2910 | 32 | 2 | 1592 |
| uci | 541909 | 343799 | 656 | 12 | 3465 | 1618 | 1163 | 684 | 1 | 289 |

The raw row counts come from the completed audit. Development counts stop at each second calibration horizon. Only 12 entities per dataset are panelled; exclusion and eligibility files retain the cohort decision record.

## Tables 2–4 — Six-model validation accuracy

Primary tables use exactly the same successful, scorable cells across all six models within each dataset. Own-successful scores and full coverage are also saved so failure exclusions remain visible. MAPE excludes actual zero demand; percentages are multiplied by 100. sMAPE assigns zero to both-zero cells. WAPE is undefined for a zero total actual. Bias is prediction minus actual. Favorita NWRMSLE is sqrt(sum(weight × squared log1p error) / sum(weight)), with weight 1.25 for perishable items and 1 otherwise. WMAE uses the same weights; it is not assigned to datasets with no justified weights.

### dunnhumby
| model | n | MAE | RMSE | MAPE | MAPE_coverage | sMAPE | WAPE | Bias | WMAE | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 168 | 0.9821 | 1.7542 | 88.7798 | 0.3333 | 92.4495 | 151.3761 | 0.1012 | — | — |
| SARIMA | 168 | 0.7571 | 1.1029 | 58.3793 | 0.3333 | 162.7224 | 116.6868 | -0.0815 | — | — |
| XGBoost | 168 | 0.7764 | 1.1310 | 56.3308 | 0.3333 | 161.6883 | 119.6582 | -0.0692 | — | — |
| LSTM | 168 | 0.6830 | 1.2883 | 91.9811 | 0.3333 | 120.2771 | 105.2730 | -0.5281 | — | — |
| Transformer | 168 | 0.7318 | 1.2195 | 66.9199 | 0.3333 | 139.2927 | 112.7854 | -0.2564 | — | — |
| Hybrid | 168 | 0.7564 | 1.1675 | 67.9860 | 0.3333 | 169.1593 | 116.5854 | -0.1890 | — | — |

### favorita
| model | n | MAE | RMSE | MAPE | MAPE_coverage | sMAPE | WAPE | Bias | WMAE | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 192 | 2.3594 | 4.0434 | 70.2750 | 0.5938 | 78.6870 | 65.2738 | -0.0885 | 2.3811 | 0.7154 |
| SARIMA | 192 | 2.8708 | 4.3665 | 65.0167 | 0.5938 | 112.0817 | 79.4239 | 0.8980 | 2.9823 | 0.9353 |
| XGBoost | 192 | 2.6661 | 4.1104 | 62.6535 | 0.5938 | 110.7672 | 73.7609 | 0.6198 | 2.7364 | 0.8453 |
| LSTM | 192 | 2.8354 | 4.3212 | 60.3865 | 0.5938 | 108.9861 | 78.4447 | 1.0205 | 2.9406 | 0.9413 |
| Transformer | 192 | 3.0814 | 4.6504 | 68.0560 | 0.5938 | 111.1245 | 85.2479 | 1.2340 | 3.1846 | 0.9778 |
| Hybrid | 192 | 2.6418 | 4.0081 | 59.9156 | 0.5938 | 110.5079 | 73.0882 | 0.7313 | 2.7210 | 0.8648 |

### uci
| model | n | MAE | RMSE | MAPE | MAPE_coverage | sMAPE | WAPE | Bias | WMAE | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 144 | 16.0694 | 41.9245 | 176.0595 | 0.5625 | 122.5289 | 161.3668 | 2.4306 | — | — |
| SARIMA | 144 | 12.2215 | 27.6816 | 172.5629 | 0.5625 | 133.4313 | 122.7265 | 0.7287 | — | — |
| XGBoost | 144 | 12.0123 | 28.8888 | 125.8842 | 0.5625 | 136.1871 | 120.6260 | -1.3702 | — | — |
| LSTM | 144 | 10.3044 | 29.5514 | 85.5478 | 0.5625 | 151.9187 | 103.4754 | -7.9012 | — | — |
| Transformer | 144 | 10.7942 | 28.4141 | 115.5536 | 0.5625 | 139.4436 | 108.3935 | -5.1456 | — | — |
| Hybrid | 144 | 11.0417 | 27.3295 | 103.1908 | 0.5625 | 138.4325 | 110.8789 | -3.0430 | — | — |

## Table 5 — Hybrid improvement over each component

Positive values favor Hybrid; negative values are deterioration. All requested metric comparisons are in `hybrid_improvements.csv`. MAE comparisons:

| dataset | baseline | absolute_improvement | percentage_improvement |
| --- | --- | --- | --- |
| dunnhumby | Naive | 0.2257 | 22.9830 |
| dunnhumby | SARIMA | 0.0007 | 0.0869 |
| dunnhumby | XGBoost | 0.0199 | 2.5680 |
| dunnhumby | LSTM | -0.0734 | -10.7458 |
| dunnhumby | Transformer | -0.0247 | -3.3692 |
| favorita | Naive | -0.2825 | -11.9717 |
| favorita | SARIMA | 0.2290 | 7.9771 |
| favorita | XGBoost | 0.0243 | 0.9120 |
| favorita | LSTM | 0.1936 | 6.8284 |
| favorita | Transformer | 0.4395 | 14.2639 |
| uci | Naive | 5.0277 | 31.2876 |
| uci | SARIMA | 1.1798 | 9.6536 |
| uci | XGBoost | 0.9706 | 8.0804 |
| uci | LSTM | -0.7373 | -7.1549 |
| uci | Transformer | -0.2475 | -2.2930 |

## Table 6 — Hybrid error by demand regime

Regimes are computed from history available before the first calibration origin. Intermittent means at most half of scorable history days have positive demand. Volatility is training coefficient-of-variation rank within the 12-entity cohort. Weekly association means training lag-7 correlation at least 0.3; this is a diagnostic threshold, not a causal or statistical seasonality test.

| dataset | dimension | segment | n | MAE | RMSE | sMAPE | WAPE | Bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | volume | high | 56 | 1.1714 | 1.6399 | 146.1699 | 99.3879 | -0.4378 |
| dunnhumby | volume | low | 56 | 0.4240 | 0.5190 | 181.5475 | 197.8744 | 0.1289 |
| dunnhumby | volume | medium | 56 | 0.6739 | 1.0634 | 179.7606 | 121.7326 | -0.2581 |
| dunnhumby | intermittency | frequent | 28 | 1.0028 | 1.3916 | 126.7281 | 85.0828 | -0.4574 |
| dunnhumby | intermittency | intermittent | 140 | 0.7071 | 1.1174 | 177.6456 | 130.2641 | -0.1353 |
| dunnhumby | volatility | high | 56 | 0.6034 | 0.9185 | 181.7709 | 153.5951 | -0.0193 |
| dunnhumby | volatility | low | 56 | 1.0608 | 1.5518 | 148.1074 | 97.3866 | -0.4123 |
| dunnhumby | volatility | medium | 56 | 0.6050 | 0.9153 | 177.5996 | 130.3126 | -0.1355 |
| favorita | volume | high | 64 | 5.4105 | 6.5031 | 92.4196 | 68.7052 | 1.9101 |
| favorita | volume | low | 64 | 1.0392 | 1.6046 | 132.7517 | 92.3691 | -0.0707 |
| favorita | volume | medium | 64 | 1.4758 | 1.8246 | 106.3522 | 80.0441 | 0.3544 |
| favorita | intermittency | frequent | 128 | 3.5293 | 4.8485 | 91.4533 | 68.7601 | 0.9352 |
| favorita | intermittency | intermittent | 64 | 0.8668 | 1.0857 | 148.6169 | 149.9401 | 0.3234 |
| favorita | volatility | high | 64 | 0.8869 | 1.1112 | 142.5386 | 135.1481 | 0.5187 |
| favorita | volatility | low | 64 | 4.5150 | 6.0082 | 58.4335 | 51.3249 | 0.4986 |
| favorita | volatility | medium | 64 | 2.5236 | 3.2956 | 130.5516 | 181.4726 | 1.1765 |
| uci | volume | high | 48 | 15.4024 | 31.0723 | 122.6934 | 118.2906 | 0.4598 |
| uci | volume | low | 48 | 5.5031 | 11.6343 | 144.6614 | 110.5232 | -2.0970 |
| uci | volume | medium | 48 | 12.2195 | 33.7617 | 147.9427 | 102.9013 | -7.4920 |
| uci | intermittency | frequent | 108 | 10.3035 | 21.8437 | 134.2142 | 113.6645 | -0.9030 |
| uci | intermittency | intermittent | 36 | 13.2563 | 39.4482 | 151.0871 | 104.8853 | -9.4631 |
| uci | volatility | high | 48 | 9.5126 | 32.7885 | 140.4051 | 106.9336 | -5.5585 |
| uci | volatility | low | 48 | 15.8408 | 31.3280 | 128.8511 | 113.4865 | -1.2309 |
| uci | volatility | medium | 48 | 7.7716 | 13.5710 | 146.0412 | 110.6938 | -2.3397 |

All six models are broken down by horizon, volume, intermittency, volatility, weekly association, recorded promotion/discount status, category, store where applicable, dataset and origin in `error_analysis.csv`. Future realized promotion and holiday fields are used only for retrospective diagnostics; forecast features contain past values and deterministic calendar terms.

### Zero-heavy scoring and bias

Dunnhumby has 112 zero actuals among 168 scored cells. Hybrid predicts positive values in every cell, while Naive is exactly correct at zero on 72 cells. Under the declared sMAPE convention, a positive forecast against zero actual has 200% error, explaining why improved MAE can coexist with worse sMAPE. UCI also has many zeros. The five-epoch neural models can have substantial negative bias and clipped outputs; lower MAE alone does not establish adequate fit. Clipping counts, bias, RMSE and MAPE coverage must be read together.

## Table 7 — One declared correction trial

The first calibration fold supplies a pooled Hybrid residual offset, which is added to second-fold forecasts and clipped at zero. This is a correction of Hybrid, not a seventh forecasting architecture. The offset is estimated before the evaluated fold. The unchanged Hybrid remains the primary result; no correction is retained for final testing on evidence from this single origin.

| dataset | offset | before | after | absolute_improvement | retained |
| --- | --- | --- | --- | --- | --- |
| dunnhumby | 0.2596 | 0.7564 | 0.8495 | -0.0931 | False |
| favorita | 0.3564 | 2.6418 | 2.7634 | -0.1216 | False |
| uci | 1.0683 | 11.0417 | 11.3729 | -0.3312 | False |

All requested before/after metrics, reasons and overfitting risks are in `error_correction_before_after.csv`. This trial reuses weight-calibration labels and may overfit that short fold. Further development origins are required before choosing a correction.

## Table 8 — Hybrid ablation

Every subset learns inverse-MAE weights on the same all-five successful calibration cells and is scored on the same all-six successful evaluation cells. Components are reused without retraining. These are exploratory diagnostics, not a selection of a new final architecture.

| dataset | combination | MAE | RMSE | sMAPE | WAPE | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Naive + SARIMA | 0.8419 | 1.2653 | 166.0965 | 129.7670 | — |
| dunnhumby | SARIMA + XGBoost | 0.7665 | 1.1141 | 162.0387 | 118.1327 | — |
| dunnhumby | XGBoost + LSTM | 0.7245 | 1.1898 | 174.9662 | 111.6596 | — |
| dunnhumby | LSTM + Transformer | 0.7052 | 1.2379 | 161.8994 | 108.6838 | — |
| dunnhumby | Full Hybrid | 0.7564 | 1.1675 | 169.1593 | 116.5854 | — |
| dunnhumby | Without Naive | 0.7309 | 1.1637 | 169.9649 | 112.6502 | — |
| dunnhumby | Without SARIMA | 0.7586 | 1.1927 | 171.3678 | 116.9158 | — |
| dunnhumby | Without XGBoost | 0.7528 | 1.1863 | 171.5277 | 116.0297 | — |
| dunnhumby | Without LSTM | 0.7843 | 1.1638 | 164.7091 | 120.8769 | — |
| dunnhumby | Without Transformer | 0.7649 | 1.1694 | 169.4651 | 117.8999 | — |
| favorita | Naive + SARIMA | 2.4850 | 3.8091 | 115.4415 | 68.7499 | 0.7855 |
| favorita | SARIMA + XGBoost | 2.7614 | 4.2028 | 110.7798 | 76.3953 | 0.8936 |
| favorita | XGBoost + LSTM | 2.7328 | 4.1738 | 109.1064 | 75.6047 | 0.8940 |
| favorita | LSTM + Transformer | 2.9511 | 4.4673 | 109.7761 | 81.6437 | 0.9594 |
| favorita | Full Hybrid | 2.6418 | 4.0081 | 110.5079 | 73.0882 | 0.8648 |
| favorita | Without Naive | 2.8447 | 4.3065 | 109.7487 | 78.6993 | 0.9258 |
| favorita | Without SARIMA | 2.6013 | 3.9481 | 110.8547 | 71.9676 | 0.8461 |
| favorita | Without XGBoost | 2.6536 | 4.0060 | 111.1111 | 73.4123 | 0.8711 |
| favorita | Without LSTM | 2.6149 | 3.9670 | 111.3578 | 72.3427 | 0.8461 |
| favorita | Without Transformer | 2.5576 | 3.8978 | 110.9229 | 70.7576 | 0.8347 |
| uci | Naive + SARIMA | 13.3329 | 29.6615 | 137.0770 | 133.8873 | — |
| uci | SARIMA + XGBoost | 12.0117 | 28.0129 | 133.9906 | 120.6194 | — |
| uci | XGBoost + LSTM | 10.6354 | 28.6817 | 141.3801 | 106.7995 | — |
| uci | LSTM + Transformer | 10.3597 | 28.8403 | 150.5069 | 104.0302 | — |
| uci | Full Hybrid | 11.0417 | 27.3295 | 138.4325 | 110.8789 | — |
| uci | Without Naive | 10.7580 | 28.1014 | 138.2398 | 108.0301 | — |
| uci | Without SARIMA | 10.9196 | 27.3904 | 141.0686 | 109.6528 | — |
| uci | Without XGBoost | 10.8889 | 27.3345 | 139.6040 | 109.3447 | — |
| uci | Without LSTM | 11.6332 | 27.1529 | 136.8917 | 116.8191 | — |
| uci | Without Transformer | 11.3311 | 27.4442 | 138.0171 | 113.7848 | — |

### Weights and error diversity

| dataset | model | weight | first_calibration_MAE |
| --- | --- | --- | --- |
| dunnhumby | Naive | 0.1634 | 1.0179 |
| dunnhumby | SARIMA | 0.1961 | 0.8480 |
| dunnhumby | XGBoost | 0.1975 | 0.8421 |
| dunnhumby | LSTM | 0.2293 | 0.7252 |
| dunnhumby | Transformer | 0.2138 | 0.7777 |
| favorita | Naive | 0.2039 | 2.8594 |
| favorita | SARIMA | 0.2092 | 2.7866 |
| favorita | XGBoost | 0.1987 | 2.9335 |
| favorita | LSTM | 0.1911 | 3.0499 |
| favorita | Transformer | 0.1972 | 2.9565 |
| uci | Naive | 0.1313 | 15.0909 |
| uci | SARIMA | 0.1794 | 11.0424 |
| uci | XGBoost | 0.1988 | 9.9621 |
| uci | LSTM | 0.2527 | 7.8384 |
| uci | Transformer | 0.2378 | 8.3299 |

| dataset | model_1 | model_2 | error_Pearson_r |
| --- | --- | --- | --- |
| dunnhumby | Naive | SARIMA | 0.6361 |
| dunnhumby | Naive | XGBoost | 0.6306 |
| dunnhumby | Naive | LSTM | 0.5988 |
| dunnhumby | Naive | Transformer | 0.6084 |
| dunnhumby | SARIMA | XGBoost | 0.9900 |
| dunnhumby | SARIMA | LSTM | 0.9713 |
| dunnhumby | SARIMA | Transformer | 0.9549 |
| dunnhumby | XGBoost | LSTM | 0.9769 |
| dunnhumby | XGBoost | Transformer | 0.9598 |
| dunnhumby | LSTM | Transformer | 0.9636 |
| favorita | Naive | SARIMA | 0.6573 |
| favorita | Naive | XGBoost | 0.7029 |
| favorita | Naive | LSTM | 0.6374 |
| favorita | Naive | Transformer | 0.6247 |
| favorita | SARIMA | XGBoost | 0.9637 |
| favorita | SARIMA | LSTM | 0.9710 |
| favorita | SARIMA | Transformer | 0.9653 |
| favorita | XGBoost | LSTM | 0.9646 |
| favorita | XGBoost | Transformer | 0.9578 |
| favorita | LSTM | Transformer | 0.9804 |
| uci | Naive | SARIMA | 0.5478 |
| uci | Naive | XGBoost | 0.4561 |
| uci | Naive | LSTM | 0.4294 |
| uci | Naive | Transformer | 0.4135 |
| uci | SARIMA | XGBoost | 0.9598 |
| uci | SARIMA | LSTM | 0.9739 |
| uci | SARIMA | Transformer | 0.9660 |
| uci | XGBoost | LSTM | 0.9431 |
| uci | XGBoost | Transformer | 0.9376 |
| uci | LSTM | Transformer | 0.9809 |

Positive residual correlations limit diversification. A convex average need not beat the best component: inverse marginal MAE ignores error covariance, regime variation and changes between folds. Weight calibration is short and noisy. Algebraic component contributions are saved separately; they should not be interpreted as causal benefits.

## Table 9 — Computational performance

Times are measured fit/predict wall times, excluding preprocessing, feature construction and serialization. Inference timings are single batch measurements, not repeated latency percentiles. RSS is sampled every 20 ms in a shared process and includes library and allocator memory. GPU memory is not applicable (CPU). Parameter counts are heterogeneous: tree nodes and neural weights are not comparable.

| dataset | model | training_seconds | inference_seconds | peak_RSS_bytes | model_bytes | parameters | parameter_kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Naive | 0.0090 | 0.0009 | 481378304 | 2796 | 0.0000 | learned_coefficients |
| dunnhumby | SARIMA | 0.9746 | 0.0162 | 555892736 | 80952584 | 48.0000 | fitted_coefficients |
| dunnhumby | XGBoost | 0.0937 | 0.0006 | 560742400 | 117785 | 1494.0000 | tree_nodes_not_neural_parameters |
| dunnhumby | LSTM | 0.3833 | 0.0009 | 562364416 | 32457 | 6958.0000 | trainable_parameters |
| dunnhumby | Transformer | 0.5369 | 0.0006 | 562511872 | 53461 | 10318.0000 | trainable_parameters |
| dunnhumby | Hybrid | 4.2506 | 0.0205 | 568033280 | 81159693 | — | heterogeneous component parameters plus five scalar weights |
| favorita | Naive | 0.0130 | 0.0010 | 707690496 | 2924 | 0.0000 | learned_coefficients |
| favorita | SARIMA | 2.5297 | 0.0167 | 744341504 | 131951569 | 48.0000 | fitted_coefficients |
| favorita | XGBoost | 0.1636 | 0.0006 | 759595008 | 116359 | 1500.0000 | tree_nodes_not_neural_parameters |
| favorita | LSTM | 0.6437 | 0.0010 | 756662272 | 33788 | 7280.0000 | trainable_parameters |
| favorita | Transformer | 0.9407 | 0.0006 | 756662272 | 54781 | 10640.0000 | trainable_parameters |
| favorita | Hybrid | 8.7675 | 0.0213 | 759595008 | 132160022 | — | heterogeneous component parameters plus five scalar weights |
| uci | Naive | 0.0067 | 0.0009 | 517947392 | 2817 | 0.0000 | learned_coefficients |
| uci | SARIMA | 0.9404 | 0.0153 | 519143424 | 38465984 | 48.0000 | fitted_coefficients |
| uci | XGBoost | 0.0784 | 0.0006 | 521175040 | 117853 | 1500.0000 | tree_nodes_not_neural_parameters |
| uci | LSTM | 0.1663 | 0.0009 | 521584640 | 32379 | 6958.0000 | trainable_parameters |
| uci | Transformer | 0.2297 | 0.0007 | 521207808 | 53317 | 10318.0000 | trainable_parameters |
| uci | Hybrid | 3.0418 | 0.0200 | 521584640 | 38672969 | — | heterogeneous component parameters plus five scalar weights |

Hybrid training cost includes first-fold component fitting, second-fold refitting and weight fitting. Standalone rows show their second-fold fit; both stages are retained in the CSV for lifecycle comparisons. Hybrid inference cost includes all five components plus combination. Its reported peak RSS is the maximum component-process peak, not a simultaneous deployment benchmark.

## Table 10 — Failures and coverage

| dataset | model | scheduled_cells | scorable_actual_cells | successful_scorable_cells | failed_scorable_cells | common_six_model_cells |
| --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Naive | 168 | 168 | 168 | 0 | 168 |
| dunnhumby | SARIMA | 168 | 168 | 168 | 0 | 168 |
| dunnhumby | XGBoost | 168 | 168 | 168 | 0 | 168 |
| dunnhumby | LSTM | 168 | 168 | 168 | 0 | 168 |
| dunnhumby | Transformer | 168 | 168 | 168 | 0 | 168 |
| dunnhumby | Hybrid | 168 | 168 | 168 | 0 | 168 |
| favorita | Naive | 192 | 192 | 192 | 0 | 192 |
| favorita | SARIMA | 192 | 192 | 192 | 0 | 192 |
| favorita | XGBoost | 192 | 192 | 192 | 0 | 192 |
| favorita | LSTM | 192 | 192 | 192 | 0 | 192 |
| favorita | Transformer | 192 | 192 | 192 | 0 | 192 |
| favorita | Hybrid | 192 | 192 | 192 | 0 | 192 |
| uci | Naive | 168 | 144 | 144 | 0 | 144 |
| uci | SARIMA | 168 | 144 | 144 | 0 | 144 |
| uci | XGBoost | 168 | 144 | 144 | 0 | 144 |
| uci | LSTM | 168 | 144 | 144 | 0 | 144 |
| uci | Transformer | 168 | 144 | 144 | 0 | 144 |
| uci | Hybrid | 168 | 144 | 144 | 0 | 144 |

No component fitting or forecasting failures occurred in this pilot. The implementation marks nonconverged SARIMA forecasts missing rather than replacing them with a successful fallback. The Naive baseline has a declared last-valid fallback for missing seasonal lags; its cell count is in the computational CSV. Detailed exception and optimizer logs are retained per dataset.

## Error associations and research questions

- **dunnhumby (RQ1–3):** Hybrid MAE 0.7564, RMSE 1.1675, WAPE 116.59%. Lowest measured pilot MAE: LSTM. Hybrid MAE is lower than Naive, SARIMA, XGBoost; higher than LSTM, Transformer (RQ4–8). This is one pilot origin.
- **dunnhumby regimes and associations (RQ9–10):** highest Hybrid volume-segment MAE is high (1.1714). In its 20 largest absolute errors, common flags are intermittent_history (16/20), promotion_or_discount_recorded (8/20), recent_training_mean_shift (4/20). These flags can overlap and do not establish causes.
- **favorita (RQ1–3):** Hybrid MAE 2.6418, RMSE 4.0081, WAPE 73.09%. Lowest measured pilot MAE: Naive. Hybrid MAE is lower than SARIMA, XGBoost, LSTM, Transformer; higher than Naive (RQ4–8). This is one pilot origin.
- **favorita regimes and associations (RQ9–10):** highest Hybrid volume-segment MAE is high (5.4105). In its 20 largest absolute errors, common flags are no_prespecified_flag (9/20), promotion_or_discount_recorded (5/20), drop_vs_training_mean (5/20). These flags can overlap and do not establish causes.
- **uci (RQ1–3):** Hybrid MAE 11.0417, RMSE 27.3295, WAPE 110.88%. Lowest measured pilot MAE: LSTM. Hybrid MAE is lower than Naive, SARIMA, XGBoost; higher than LSTM, Transformer (RQ4–8). This is one pilot origin.
- **uci regimes and associations (RQ9–10):** highest Hybrid volume-segment MAE is high (15.4024). In its 20 largest absolute errors, common flags are less_than_one_year_history (20/20), recent_training_mean_shift (6/20), intermittent_history (5/20). These flags can overlap and do not establish causes.

**RQ11:** The single correction improves validation MAE on none of the datasets and worsens it on dunnhumby, favorita, uci. See Table 7 for magnitudes. It is not retained.

**RQ12:** Generalization of a corrected Hybrid across the datasets is not established. No final-test evaluation has been run.

**RQ13:** Tables 5 and 8 and the residual correlations provide measured evidence about combination on this pilot only. Combining all five is not presumed beneficial, and subset comparisons are not final model selection.

**RQ14:** Table 9 contains measured small-cohort CPU costs. They are not full-dataset runtime or memory estimates.

**RQ15:** Horizon and entity results are saved for every included cell in `error_analysis.csv` and `entity_accuracy.csv`. Consistency beyond a single evaluated origin cannot be established. Aggregate improvements must not be treated as uniform gains across horizons or products.

| dataset | dimension | baseline | Hybrid_lower_MAE | Hybrid_higher_MAE | ties | total |
| --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | entity | Naive | 9 | 3 | 0 | 12 |
| dunnhumby | entity | SARIMA | 6 | 6 | 0 | 12 |
| dunnhumby | entity | XGBoost | 9 | 3 | 0 | 12 |
| dunnhumby | entity | LSTM | 2 | 10 | 0 | 12 |
| dunnhumby | entity | Transformer | 4 | 8 | 0 | 12 |
| dunnhumby | horizon | Naive | 10 | 4 | 0 | 14 |
| dunnhumby | horizon | SARIMA | 6 | 8 | 0 | 14 |
| dunnhumby | horizon | XGBoost | 8 | 6 | 0 | 14 |
| dunnhumby | horizon | LSTM | 4 | 10 | 0 | 14 |
| dunnhumby | horizon | Transformer | 7 | 7 | 0 | 14 |
| favorita | entity | Naive | 8 | 4 | 0 | 12 |
| favorita | entity | SARIMA | 9 | 3 | 0 | 12 |
| favorita | entity | XGBoost | 7 | 5 | 0 | 12 |
| favorita | entity | LSTM | 7 | 5 | 0 | 12 |
| favorita | entity | Transformer | 10 | 2 | 0 | 12 |
| favorita | horizon | Naive | 5 | 11 | 0 | 16 |
| favorita | horizon | SARIMA | 12 | 4 | 0 | 16 |
| favorita | horizon | XGBoost | 9 | 7 | 0 | 16 |
| favorita | horizon | LSTM | 13 | 3 | 0 | 16 |
| favorita | horizon | Transformer | 15 | 1 | 0 | 16 |
| uci | entity | Naive | 8 | 4 | 0 | 12 |
| uci | entity | SARIMA | 9 | 3 | 0 | 12 |
| uci | entity | XGBoost | 8 | 4 | 0 | 12 |
| uci | entity | LSTM | 2 | 10 | 0 | 12 |
| uci | entity | Transformer | 5 | 7 | 0 | 12 |
| uci | horizon | Naive | 8 | 4 | 0 | 12 |
| uci | horizon | SARIMA | 11 | 1 | 0 | 12 |
| uci | horizon | XGBoost | 7 | 5 | 0 | 12 |
| uci | horizon | LSTM | 4 | 8 | 0 | 12 |
| uci | horizon | Transformer | 6 | 6 | 0 | 12 |

Statistical significance could not be reliably established because of the limited number of independent forecast origins. No p-values, confidence intervals or Diebold–Mariano claims are manufactured.

The diagnostics do not establish underfitting, overfitting, latent availability, causal promotion effects, or store sparsity from this small cohort. XGBoost, LSTM and Transformer generate direct horizons without recursively feeding predictions into their features. SARIMA propagates its state across the horizon; this pilot does not isolate propagation error from other error sources. Passing leakage and join checks is evidence for tested contracts, not proof against every possible data issue.

## Feasibility gate and next research stage

The exercised pipeline is feasible on the 12-entity CPU pilot. Full training remains disabled. The machine has 8 GiB RAM and only about 2.5 GiB free disk before model artifacts. The 5 GB Favorita train CSV must remain compressed and streamed. Full populations include many thousands of entities; per-entity SARIMA storage/fitting and global neural window materialization require a separately bounded design. Scaling these pilot measurements linearly would be unjustified.

Before a full experiment: provision working storage, use disk-bounded streaming training windows, set a training-only cohort and computational budget, use the unchanged tuning/calibration origins to compare declared configurations, validate any selected correction on additional untouched development origins, freeze configuration and checksums, then run the final test once. Do not promote pilot-ranked variants or adjust final-test boundaries.

## Reproducibility and saved artifacts

- `experiments/continuation_protocol.json`: declared scope, architecture, correction budget and disabled final test.
- `experiments/leakage_audit_report.json`: pre-training leakage gate and code/data checksums.
- `experiments/pilot_snapshot.json`: versions, code/configuration/prediction checksums; pilot snapshot only.
- `data/processed/pilot/`: panels, masks, eligibility/cohort/exclusions, preprocessing and returns ledgers.
- `models/pilot/`: component artifacts, transform states, configurations and frozen Hybrid weights.
- `predictions/pilot/`: both-fold forecasts and one correction trial.
- `logs/pilot/`: training histories, execution and optimizer warning logs.
- `results/pilot/results_verification.json`: metric and prediction consistency checks.
- `results/pilot/saved_model_replay.json`: all 18 saved second-fold model outputs reproduced within 1e-6.

See `experiments/pilot_reproduction.md` for commands and scope. The completed audit remains unchanged as historical context. Its earlier statement that no models had been trained describes the audit stage, not this continuation.
