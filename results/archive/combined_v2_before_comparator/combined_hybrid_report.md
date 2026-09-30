# Combined-dataset Hybrid development experiment

Three datasets are combined as a retailer-labelled panel of the unchanged 36-series cohort (12 from each source). All raw sources and the earlier pilot are preserved. This is not full-population training. The development panel contains 23,318 daily rows; it ends before each original final-test boundary.

Corrected Hybrid versus the inverse-MAE reference on later development folds: 2.35% improvement in equal-dataset normalized MAE. dunnhumby: 7.86% MAE improvement; favorita: 4.38% MAE improvement; uci: -2.27% MAE improvement. Negative improvements are deterioration. MAE and RMSE trade-offs must be read together; these results do not establish universal superiority. No evaluation-based corrections were made after seeing them.

## What changed

XGBoost, LSTM and Transformer now share training across the three sources with distinct entity IDs and domain indicators. Targets are scaled by training-only entity means, and training weights balance the three datasets. Naive and SARIMA remain local time-series models. Native-unit errors are reported separately; the combined score averages normalized MAE equally across datasets. Different historical calendars are an offline multi-domain experiment, not a claim of contemporaneous operational forecasts.

The Transformer component follows the paper’s pre-normalized temporal attention and modality-specific cross-attention idea (equations 9–12), using history, recorded promotion/price, and available external/business features. Its compact 64-dimensional, four-head architecture is an explicit CPU adaptation. Missing inventory, sensors and competitors are not fabricated. This is not the paper’s full 512-dimensional, six-layer forecasting-plus-RL architecture.

Hybrid tries four declared weighting rules: inverse MAE, shared normalized-MAE optimization, dataset-specific shrinkage, and intermittency-regime shrinkage. Optimized convex weights retain at least 1% for each of the five components. The two shrinkage constants are fixed, not searched. All trials and any deterioration are retained.

## Selection and error correction

Fold 1 supplies out-of-sample errors. Fold 2 selects the weighting rule and one correction shrinkage factor from 0, 0.5 or 1. Weights and residual offsets are then fitted using folds 1–2 only and frozen for folds 3–4. Components are refitted at each later origin using only then-available history; Hybrid parameters remain fixed. No changes are selected from later evaluation errors. The pre-existing final test is unused.

Selected Hybrid method: **regime_shrinkage**. The objective is equal-dataset mean normalized MAE, not whichever metric later looks best. Error correction adds a dataset median normalized residual, scaled back into native units; negative forecasts are clipped to zero. This is our declared extension, not a residual-correction algorithm supplied by the paper.

| candidate | fold_2_equal_domain_normalized_MAE |
| --- | --- |
| inverse_MAE | 0.8579 |
| pooled_MAE | 0.8255 |
| dataset_shrinkage | 0.8008 |
| regime_shrinkage | 0.8002 |

## Measured accuracy on later calibration folds

| dataset | model | n | MAE | RMSE | MAPE | MAPE_coverage | sMAPE | WAPE | Bias | NWRMSLE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dunnhumby | Hybrid_reference | 336 | 0.7807 | 1.3729 | 65.1500 | 0.3185 | 168.9791 | 114.0555 | -0.1913 | nan |
| dunnhumby | Hybrid_v2 | 336 | 0.7266 | 1.4875 | 82.6622 | 0.3185 | 183.8152 | 106.1517 | -0.4652 | nan |
| dunnhumby | Hybrid_v2_corrected | 336 | 0.7194 | 1.4942 | 83.9656 | 0.3185 | 151.7899 | 105.0885 | -0.4855 | nan |
| dunnhumby | LSTM | 336 | 0.7167 | 1.5353 | 89.6464 | 0.3185 | 134.1219 | 104.7068 | -0.5547 | nan |
| dunnhumby | Naive | 336 | 0.9881 | 1.8420 | 87.9907 | 0.3185 | 87.1848 | 144.3478 | 0.0893 | nan |
| dunnhumby | SARIMA | 336 | 0.8209 | 1.3559 | 58.9431 | 0.3185 | 164.8995 | 119.9244 | -0.1082 | nan |
| dunnhumby | Transformer | 336 | 0.7440 | 1.3498 | 67.5562 | 0.3185 | 150.9300 | 108.6937 | -0.2450 | nan |
| dunnhumby | XGBoost | 336 | 0.8508 | 1.3545 | 57.9010 | 0.3185 | 162.5938 | 124.2918 | -0.0261 | nan |
| favorita | Hybrid_reference | 384 | 2.1360 | 3.4208 | 60.7620 | 0.5807 | 115.4565 | 64.4310 | 0.0326 | 0.6956 |
| favorita | Hybrid_v2 | 384 | 2.0926 | 3.5468 | 64.1081 | 0.5807 | 121.7754 | 63.1245 | -0.3723 | 0.6589 |
| favorita | Hybrid_v2_corrected | 384 | 2.0423 | 3.7724 | 61.7840 | 0.5807 | 96.1337 | 61.6065 | -0.9826 | 0.6191 |
| favorita | LSTM | 384 | 2.8028 | 5.3429 | 79.2746 | 0.5807 | 122.7649 | 84.5467 | -2.4283 | 1.0438 |
| favorita | Naive | 384 | 2.2240 | 4.1022 | 89.5386 | 0.5807 | 69.5121 | 67.0856 | -0.0625 | 0.6982 |
| favorita | SARIMA | 384 | 2.6840 | 4.0080 | 71.4252 | 0.5807 | 113.1556 | 80.9639 | 1.1540 | 0.9058 |
| favorita | Transformer | 384 | 2.2744 | 3.4764 | 68.9915 | 0.5807 | 97.4053 | 68.6086 | 0.6843 | 0.7430 |
| favorita | XGBoost | 384 | 2.3438 | 3.6828 | 68.5815 | 0.5807 | 112.8015 | 70.7018 | 0.7214 | 0.7528 |
| uci | Hybrid_reference | 288 | 18.4696 | 62.1960 | 107.4292 | 0.6250 | 126.5774 | 93.6651 | -10.2521 | nan |
| uci | Hybrid_v2 | 288 | 18.7814 | 64.5008 | 90.1406 | 0.6250 | 154.0097 | 95.2464 | -15.9642 | nan |
| uci | Hybrid_v2_corrected | 288 | 18.8889 | 64.7276 | 94.1419 | 0.6250 | 154.7388 | 95.7913 | -16.5323 | nan |
| uci | LSTM | 288 | 19.2013 | 65.0899 | 100.7804 | 0.6250 | 140.5402 | 97.3758 | -17.0194 | nan |
| uci | Naive | 288 | 23.7743 | 66.4257 | 144.4227 | 0.6250 | 125.3847 | 120.5670 | -5.9132 | nan |
| uci | SARIMA | 288 | 19.3611 | 62.8315 | 143.7013 | 0.6250 | 126.8627 | 98.1861 | -8.7067 | nan |
| uci | Transformer | 288 | 18.4612 | 61.7340 | 128.0749 | 0.6250 | 134.6025 | 93.6227 | -10.5542 | nan |
| uci | XGBoost | 288 | 19.3662 | 61.7273 | 154.9163 | 0.6250 | 124.3789 | 98.2122 | -6.4440 | nan |

MAPE excludes zero actuals and reports coverage; sMAPE treats both-zero cells as zero error. Favorita NWRMSLE uses the original 1/1.25 item weights. Scoring uses identical successful/scorable cells for all compared components and Hybrids; failures and scheduled counts are saved separately.

Combined normalized scores (smaller is better):

| model | equal_dataset_mean_normalized_MAE |
| --- | --- |
| Hybrid_reference | 0.9490 |
| Hybrid_v2 | 0.9331 |
| Hybrid_v2_corrected | 0.9267 |
| LSTM | 0.9693 |
| Naive | 1.1823 |
| SARIMA | 1.0482 |
| Transformer | 0.9612 |
| XGBoost | 1.0190 |

The reference Hybrid uses the earlier inverse-MAE rule with the same newly fitted components. This isolates the weighting change; it is not a direct comparison with the earlier pilot’s different models and dates.

## Accuracy improvements and correction effects

| dataset | baseline | comparison | baseline_error | comparison_error | percentage_improvement |
| --- | --- | --- | --- | --- | --- |
| dunnhumby | Hybrid_reference | Hybrid_v2 | 0.7807 | 0.7266 | 6.9298 |
| dunnhumby | Hybrid_v2 | Hybrid_v2_corrected | 0.7266 | 0.7194 | 1.0016 |
| favorita | Hybrid_reference | Hybrid_v2 | 2.1360 | 2.0926 | 2.0278 |
| favorita | Hybrid_v2 | Hybrid_v2_corrected | 2.0926 | 2.0423 | 2.4048 |
| uci | Hybrid_reference | Hybrid_v2 | 18.4696 | 18.7814 | -1.6883 |
| uci | Hybrid_v2 | Hybrid_v2_corrected | 18.7814 | 18.8889 | -0.5721 |

Positive improvements favor the comparison method; negative values are deterioration. All metrics and every correction trial are in the accompanying CSVs. A zero selected factor means correction was rejected during development. RMSE and MAE can favor different choices.

## MARIOD comparison boundary

**MARIOD accuracy and correction comparison is pending the user’s reproduction-scope choice.** The supplied data lack inventories, sensors, supply-chain topology and operating costs required by full MARIOD. The user was asked whether to authorize a clearly labelled forecasting-only reconstruction or supply the authors’ implementation and additional data. No made-up MARIOD predictions or accuracy numbers are reported.

The paper’s Table 2 numbers are not comparable with this cohort and evaluation: they do not identify dataset-specific prediction rows, horizons or scoring population. They are not used as target scores to beat. See `paper_reproduction_assessment.md` for the exact omissions and architecture mapping.

## Limitations and reproducibility

Only two later evaluation origins per source are available, so reliable statistical superiority or full-population generalization is not established. Sparse demand, conditional zero filling, short history and unavailable business modalities remain substantive limitations. No p-values are manufactured. The experiment uses actual observed purchase/sales targets, not reconstructed latent demand or synthetic inventories.

A full-population run and full MARIOD reproduction remain separate, unfinished work. The paper reports an eight-A100 training setup; this is a bounded CPU experiment. Full inventory error/stockout comparisons are impossible from the available data alone.

Code: `src/combined_v2/`; protocol, data manifest and leakage gate: `experiments/combined_v2/`; model states: `models/combined_v2/`; all folds, metrics, correction trials, segments and measured resource costs: `results/combined_v2/`. The previous pilot remains unchanged.
