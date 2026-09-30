# Frozen final-test comparison

**Primary result:** corrected Hybrid has 2.54% lower equal-dataset normalized MAE than the corrected compact MARIOD-style forecasting reconstruction (0.782771 versus 0.803169). This does not establish superiority over full MARIOD or over the complete dataset populations.

## What was completed

Forecasting-only comparator and its residual correction; configuration/checksum freeze before test preprocessing; pre-test leakage audit; one refit of all five components using pre-test observations; forecasts across all four original final-test origins with fixed parameters; common-population accuracy, before/after correction, failure coverage, and horizon/entity/origin analysis. Both methods have the same three-factor correction-selection budget on calibration folds 1–2. No correction or architecture was chosen using final-test errors.

This remains the original 36-series cohort: 12 per dataset selected from initial-training information. Full-population training was not performed. Final test dates: Dunnhumby DAY 656–711, Favorita 2017-06-13–2017-08-15, UCI 2011-10-14–2011-12-08. Each source retains its original time axis.

## Primary accuracy comparison

| dataset | model | n | MAE | RMSE | MAPE | MAPE_coverage | sMAPE | WAPE | Bias | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Corrected Hybrid | 672 | 0.6409 | 1.3096 | 81.0939 | 0.3185 | 149.5021 | 102.7844 | -0.4145 | nan |
| dunnhumby | Corrected MARIOD-style forecaster | 672 | 0.5987 | 1.2604 | 79.7824 | 0.3185 | 94.2044 | 96.0209 | -0.4094 | nan |
| favorita | Corrected Hybrid | 768 | 2.5322 | 9.1862 | 65.8134 | 0.5430 | 96.2322 | 77.0205 | -0.5028 | 0.7138 |
| favorita | Corrected MARIOD-style forecaster | 768 | 2.4608 | 7.9925 | 63.1717 | 0.5430 | 92.9744 | 74.8461 | -1.4125 | 0.7024 |
| uci | Corrected Hybrid | 576 | 13.8462 | 33.3778 | 130.8082 | 0.6389 | 136.6514 | 104.4447 | -5.8139 | nan |
| uci | Corrected MARIOD-style forecaster | 576 | 16.2191 | 34.5847 | 182.8046 | 0.6389 | 129.0500 | 122.3444 | -1.8287 | nan |

Per-dataset MAE gains (positive favors Hybrid; negative is deterioration):

| dataset | baseline_error | comparison_error | percentage_improvement |
| --- | --- | --- | --- |
| dunnhumby | 0.5987 | 0.6409 | -7.0438 |
| favorita | 2.4608 | 2.5322 | -2.9052 |
| uci | 16.2191 | 13.8462 | 14.6306 |

## Error correction before and after

| dataset | baseline | baseline_error | comparison_error | percentage_improvement |
| --- | --- | --- | --- | --- |
| dunnhumby | Hybrid_v2 | 0.6476 | 0.6409 | 1.0314 |
| dunnhumby | MARIOD_forecasting_reconstruction | 0.6531 | 0.5987 | 8.3250 |
| favorita | Hybrid_v2 | 2.6249 | 2.5322 | 3.5287 |
| favorita | MARIOD_forecasting_reconstruction | 2.2959 | 2.4608 | -7.1820 |
| uci | Hybrid_v2 | 13.9680 | 13.8462 | 0.8720 |
| uci | MARIOD_forecasting_reconstruction | 17.7361 | 16.2191 | 8.5529 |

Corrections add a frozen dataset-specific median normalized residual multiplied by the fixed pre-test scale and selected shrinkage, then clip to nonnegative demand. The paper does not supply this residual-correction method: it is an explicitly shared extension. Correction may improve one metric while worsening another. Full MAE/RMSE/sMAPE/WAPE and Favorita NWRMSLE comparisons are in `accuracy_and_correction_comparisons.csv`.

## All model scores

| dataset | model | MAE | RMSE | sMAPE | WAPE | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Hybrid_reference | 0.7026 | 1.2057 | 168.7411 | 112.6870 | nan |
| dunnhumby | Hybrid_v2 | 0.6476 | 1.3027 | 180.8743 | 103.8555 | nan |
| dunnhumby | Hybrid_v2_corrected | 0.6409 | 1.3096 | 149.5021 | 102.7844 | nan |
| dunnhumby | LSTM | 0.6502 | 1.3556 | 139.7794 | 104.2834 | nan |
| dunnhumby | MARIOD_forecasting_reconstruction | 0.6531 | 1.2018 | 146.9472 | 104.7405 | nan |
| dunnhumby | MARIOD_forecasting_reconstruction_corrected | 0.5987 | 1.2604 | 94.2044 | 96.0209 | nan |
| dunnhumby | Naive | 0.8795 | 1.7048 | 78.2057 | 141.0501 | nan |
| dunnhumby | SARIMA | 0.7555 | 1.2120 | 164.0311 | 121.1703 | nan |
| dunnhumby | Transformer | 0.6531 | 1.2018 | 146.9472 | 104.7405 | nan |
| dunnhumby | XGBoost | 0.7650 | 1.1977 | 161.7618 | 122.6882 | nan |
| favorita | Hybrid_reference | 2.5303 | 8.3446 | 118.5878 | 76.9610 | 0.7807 |
| favorita | Hybrid_v2 | 2.6249 | 9.1555 | 123.7469 | 79.8377 | 0.7662 |
| favorita | Hybrid_v2_corrected | 2.5322 | 9.1862 | 96.2322 | 77.0205 | 0.7138 |
| favorita | LSTM | 2.7917 | 8.5617 | 108.4524 | 84.9113 | 0.9169 |
| favorita | MARIOD_forecasting_reconstruction | 2.2959 | 7.6014 | 90.2894 | 69.8308 | 0.6710 |
| favorita | MARIOD_forecasting_reconstruction_corrected | 2.4608 | 7.9925 | 92.9744 | 74.8461 | 0.7024 |
| favorita | Naive | 3.1940 | 12.8920 | 70.9953 | 97.1485 | 0.8285 |
| favorita | SARIMA | 2.9995 | 8.1052 | 117.3692 | 91.2308 | 0.9842 |
| favorita | Transformer | 2.2959 | 7.6014 | 90.2894 | 69.8308 | 0.6710 |
| favorita | XGBoost | 2.6363 | 7.9110 | 117.6402 | 80.1846 | 0.8494 |
| uci | Hybrid_reference | 15.1158 | 33.3393 | 126.4160 | 114.0221 | nan |
| uci | Hybrid_v2 | 13.9680 | 33.2437 | 135.8487 | 105.3635 | nan |
| uci | Hybrid_v2_corrected | 13.8462 | 33.3778 | 136.6514 | 104.4447 | nan |
| uci | LSTM | 13.9826 | 33.8297 | 130.9471 | 105.4735 | nan |
| uci | MARIOD_forecasting_reconstruction | 17.7361 | 35.2102 | 129.8074 | 133.7871 | nan |
| uci | MARIOD_forecasting_reconstruction_corrected | 16.2191 | 34.5847 | 129.0500 | 122.3444 | nan |
| uci | Naive | 21.6562 | 67.3415 | 118.5393 | 163.3578 | nan |
| uci | SARIMA | 14.9516 | 32.7824 | 127.4234 | 112.7835 | nan |
| uci | Transformer | 17.7361 | 35.2102 | 129.8074 | 133.7871 | nan |
| uci | XGBoost | 15.1801 | 32.1232 | 125.7675 | 114.5067 | nan |

Combined normalized scores (mean of each dataset’s mean absolute error divided by each series’s frozen pre-test mean scale, floored at one):

| model | equal_dataset_normalized_MAE |
| --- | --- |
| Hybrid_reference | 0.8334 |
| Hybrid_v2 | 0.7966 |
| Hybrid_v2_corrected | 0.7828 |
| LSTM | 0.8085 |
| MARIOD_forecasting_reconstruction | 0.8559 |
| MARIOD_forecasting_reconstruction_corrected | 0.8032 |
| Naive | 1.1112 |
| SARIMA | 0.9074 |
| Transformer | 0.8559 |
| XGBoost | 0.8866 |

Raw demand units are not summed across unrelated retailers. MAPE excludes zero actuals, sMAPE assigns zero to both-zero cells, and Favorita NWRMSLE uses perishable weights 1.25 versus 1. Primary scores use the exact same successful/scorable cells for all five components; own-successful metrics and failure counts are also saved. No missing forecast is silently replaced.

## Consistency and statistical limits

| dataset | dimension | Hybrid_lower_MAE | Hybrid_higher_MAE | ties | total |
| --- | --- | --- | --- | --- | --- |
| dunnhumby | fold | 0 | 4 | 0 | 4 |
| dunnhumby | horizon | 3 | 11 | 0 | 14 |
| dunnhumby | entity | 1 | 11 | 0 | 12 |
| favorita | fold | 2 | 2 | 0 | 4 |
| favorita | horizon | 8 | 8 | 0 | 16 |
| favorita | entity | 7 | 5 | 0 | 12 |
| uci | fold | 3 | 1 | 0 | 4 |
| uci | horizon | 8 | 4 | 0 | 12 |
| uci | entity | 7 | 5 | 0 | 12 |

Only four forecast-origin blocks per source are available. Reliable statistical significance or full-population superiority is not established; individual rows and products are not treated as independent trials. No p-values are manufactured. The direction of error improvement must be read by dataset, metric and origin.

## Frozen-model integrity

Neural/tree parameters, transforms, Hybrid weights and residual offsets stay fixed throughout testing. Later origins may use observations already available by that origin. SARIMA filtering state is refreshed with fixed coefficients (`refit=False`); coefficients are checked for exact equality. Saved-model replay reproduces first-origin forecasts within 1e-6, and all artifact checksums remain unchanged.

The comparator is exactly the Hybrid’s compact cross-modal Transformer component, with its own separately calibrated correction. It uses the same data and forecasting features; it is not an independent full-MARIOD implementation. The paper’s hierarchical inventory agents, sensor streams and inventory optimization are unavailable. Published paper metrics use a different unspecified scoring population and are not ranked against these results.

## Resources and unfinished full-scale work

Final component fitting took 68.56 seconds in total (excluding feature preparation and serialization); summed component forecasting/filtering time across all test origins was 0.681 seconds. Hybrid deployment needs all five components; the forecasting comparator needs only Transformer. Calibration/tuning costs are additional. Component timings and memory are recorded in `fold_1_compute.csv` and `test_inference_costs.csv`.

The training-only eligibility lists contain 151,723 entities. The current full-population feature materialization would require approximately 173.4 GiB for feature arrays alone, before models and copies, versus this machine’s 8 GiB RAM and roughly 4.7 GiB free disk at assessment. No unsafe full-population run was launched. It needs an out-of-core implementation and additional compute/storage; this is a real unfinished task, not completed by subsampling.

Full MARIOD additionally needs the authors’ implementation or fully specified reconstruction plus real inventory/sensor/graph/cost inputs. Those inputs have not been supplied. Consequently the answer to “is my Hybrid better than full MARIOD?” remains unproven, regardless of this forecasting-only result.

All prior experiments are preserved. Final-test results are final evidence for these frozen choices; they must not be used to retune and report this same test as untouched again. See `experiments/final_experiment/README.md` for artifact locations and reproduction rules.
