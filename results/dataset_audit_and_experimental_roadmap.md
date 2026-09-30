# A. Dataset acquisition status

Audit completed on 2026-09-22. All three requested datasets have been accessed and fully scanned for the reported raw-data statistics. **No forecasting model has been trained.** The experimental design below is proposed; there are no accuracy results, learned Hybrid weights or simulated inventory results.

| Dataset | Acquisition outcome | Version and provenance limitation |
|---|---|---|
| Dunnhumby – The Complete Journey | Public full CSV archive obtained from a [GitHub release](https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data). | [Publisher](https://www.dunnhumby.com/source-files/) requires a registration/terms form. It was not submitted. Mirror checksum matches its release digest; identity with publisher bytes or MARIOD's processed version is unverified. |
| Corporación Favorita Grocery Sales | Your supplied ZIP was copied into the project and its original item-level files audited. | [Official Kaggle source](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data). Direct unauthenticated API access returned 401; your archive resolved access. This is the original competition, not the later family-level Store Sales dataset. |
| UCI Online Retail | Downloaded the original XLSX directly from [UCI](https://archive.ics.uci.edu/dataset/352/online+retail). | The date/row coverage matches the named source, but identity with MARIOD's cleaned input is not established. Source license: CC BY 4.0. |

Original archives are preserved in `data/raw`. SHA-256 hashes and sizes for every retained raw file are in `data/profiles/raw_file_manifest.json`; acquisition details are in `source_manifest.json`. Favorita train expands to **4,997,452,288 bytes**. Low disk space interrupted an initial extraction; the incomplete generated CSV was removed and the entire train file was subsequently streamed successfully from its 7z archive. Your original Downloads ZIP and the project archive copy are preserved. No additional retail dataset was used.

# B. Dataset profiles

| Dataset | Raw rows | Columns | Coverage | Stores | Observed products | Customers |
|---|---:|---:|---|---:|---:|---:|
| Dunnhumby | 2,595,732 | 12 | DAY 1–711 (calendar anchor unknown) | 582 | 92,339 | 2,500 |
| Favorita | 125,497,040 | 6 | 2013-01-01 to 2017-08-15 | 54 | 4,036 | not supplied |
| UCI Online Retail | 541,909 | 8 | 2010-12-01 to 2011-12-09 | online channel | 4,070 | 4,372 |


Rows count transaction **lines** for Dunnhumby/UCI and daily store/item records for Favorita. They are not comparable counts of shopping trips. Product counts above come from observed sales records, not the product master. Missing customer IDs are not counted as an extra customer. Raw counts precede any proposed filtering.

| Profile detail | Dunnhumby | Favorita | UCI |
|---|---|---|---|
| Main file size | 141,742,346 uncompressed bytes | 4,997,452,288 uncompressed bytes; 474,092,593-byte train 7z | 23,715,344-byte XLSX |
| Archive size | 135,859,064 bytes | 480,014,675 bytes | Preserved ZIP; exact bytes in manifest |
| Time representation | DAY 1–711; WEEK_NO 1–102 | 1,688 calendar days, 1,684 with train records | 374 calendar days, 305 with records; timestamp endpoints 08:26 and 12:50 |
| Trips / invoices | 276,484 baskets | No transaction IDs per sales row; store-level counts in transactions.csv | 25,900 invoice identifiers |
| Catalog / categories | 92,353 product-master rows; 44 department labels, 308 commodities, 2,383 subcommodities including blank labels | 4,100 item-master rows; 33 families, 337 classes | No category master; 650 StockCodes have multiple descriptions |
| Customer coverage | 2,500 households; demographics for 801 only | No household/customer identifiers | 4,372 known customers; 38 destination countries |
| Prices | Realized sales value and discounts; no explicit shelf-price field | No item price field | UnitPrice in GBP |
| Promotions | Causal display/mailer, campaigns, coupons, redemption tables | onpromotion plus holiday/event calendar | No explicit promotion field |
| External variables | Selected household demographics; no verified calendar anchor | Store metadata, holidays, oil, daily store transaction counts | Country; no supplied holiday/economic series |
| Actual inventory | Absent | Absent | Absent |

Machine-readable full profiles: `data/profiles/dataset_profile.json`, with separate per-dataset JSON files, schema, per-column null/distinct counts and audit methods. Supporting table sizes:

| Dataset | Table | Rows | Columns |
|---|---|---:|---:|
| Dunnhumby | campaign_desc.csv | 30 | 4 |
| Dunnhumby | campaign_table.csv | 7,208 | 3 |
| Dunnhumby | coupon.csv | 124,548 | 3 |
| Dunnhumby | coupon_redempt.csv | 2,318 | 4 |
| Dunnhumby | hh_demographic.csv | 801 | 8 |
| Dunnhumby | product.csv | 92,353 | 7 |
| Dunnhumby | transaction_data.csv | 2,595,732 | 12 |
| Dunnhumby | causal_data.csv | 36,786,524 | 5 |
| Favorita | holidays_events | 350 | 6 |
| Favorita | items | 4,100 | 4 |
| Favorita | oil | 1,218 | 2 |
| Favorita | stores | 54 | 5 |
| Favorita | transactions | 83,488 | 3 |
| Favorita | test | 3,370,464 | 5 |

Favorita `test.csv` covers 2017-08-16–2017-08-31, with 3,370,464 rows and 3,901 items, and has no observed target. Its sample submission contains placeholder predictions, not actual demand, and will never be used as ground truth.

# C. Dataset comparison

Published descriptions below come from [MARIOD Section 4.1 / Table 1](https://doi.org/10.3390/s25082428). Measured columns come from this audit.

| Dataset | MARIOD description | Downloaded / supplied dataset | Differences |
|---|---|---|---|
| Dunnhumby | 2.5M records; 92 stores; about 43,000 SKUs; over 2,500 households; two years | 2,595,732 lines; 582 store IDs; 92,339 observed products; 2,500 households; DAY 1–711 | Store/product counts differ substantially; calendar dates and paper-specific inventory augmentation unavailable. |
| Favorita | 174M records; 54 stores; about 33,000 SKUs; 2013–2017 | 125,497,040 labelled records; 54 stores; 4,036 observed items; 4,100 catalog items | Store count agrees; row/product counts do not. Exact paper version remains unresolved. |
| UCI | 541,909 records; 4,373 products; online; 2010-12-01–2011-12-09 | 541,909 lines; 4,070 stock codes; 4,372 known customers; matching endpoints | Rows/dates agree; reported product count does not. Do not reinterpret customer count as product count. |

Do not silently “correct” the paper or claim a particular reason for these differences. They could involve an undisclosed subset, augmentation, version change or reporting error; the available evidence does not settle which. Sections E–M define a new, documented experiment on the acquired versions.

# D. Data-quality report

## Dunnhumby

- Transaction table: no parser-null values and no identical full rows; 14,466 zero-quantity rows, no negative quantities, 18,850 zero-sales-value rows and 4,451 positive-quantity/zero-value rows. Maximum raw quantity is 89,638. Large quantities may have different measurement units; do not declare every extreme an error or clip from whole-file quantiles.
- Product master contains whitespace-only values: 15 department, 15 commodity, 15 subcommodity and 30,607 size entries. These are semantic missing values even though CSV parsing reports zero nulls. Normalize whitespace and use explicit unknown categories. Product IDs are unique and transaction product joins have no orphans.
- There are 940,071 observed store/product pairs, indicating substantial panel sparsity. Descriptive positive-product/day aggregation finds 20,122,928 absent days inside products' first/last observed ranges. This is **not a verified count of zero-demand or stockout days** and must not define future availability.
- Demographics cover 801 of 2,500 households; 1,699 lack that table. Keep their purchases. Do not inner-join demographics and silently discard most households.
- Coupon metadata has 5,164 identical excess rows. Preserve raw entries, deduplicate the coupon/product/campaign mapping in the derived join table and record impact. Coupon redemption is a many-to-many mapping risk: aggregate exposure and redemption features before joining to demand.
- Causal table: 36,786,524 rows, 115 store IDs, 68,377 products and 93 week values. Full cross-chunk verification found zero identical rows but **15,245 excess product/store/week keys** with multiple display/mailer combinations. Aggregate to one row per join key (for example, multi-hot display/mailer indicators) before joining. Assert that sales row counts and quantity totals remain unchanged.
- `DAY` is relative; calendar origin is unverified. Do not map it to an invented year or borrow dates from a differently processed package. Validate WEEK_NO versus DAY before calendar joins. Preserve zero-quantity records for the adjustment ledger; exclude them from positive-unit demand while retaining genuine zero-demand periods on the panel. Zero-price gifts remain possible purchases and require a documented rule.

## Favorita

- Full streaming scan covered **125,497,040** train rows, not a sample. IDs are consecutive; no duplicate full rows or duplicate store/item/date keys were found. All item and store references match the masters.
- `onpromotion` has **21,657,651** missing entries (17.26% of rows). Unknown promotion is distinct from false: retain an unknown indicator; do not automatically treat the early missing block as “no promotion.” Other train columns contain no parser-null values.
- There are **7,795** negative-sales rows, **8,191,095** fractional values and no explicit zero-sales rows. Negative values range down to -15372; the maximum is 89440. Preserve fractional units. Retain negative adjustments separately and document nonnegative clipping as a target definition, not recovery of gross demand.
- There are **174,685** observed store/item pairs. The full catalog cross-product has 248,226,160 absent cells; even within each pair's first/last observations there are 63,592,025 absent dates. These include assortment and capture ambiguity. Missing rows cannot all be declared observed zero sales.
- No train records exist for Christmas Day in 2013–2016. This may reflect closure rather than missing collection, but the archive does not prove the cause. Keep these dates on the time axis and record their treatment explicitly.
- Oil has 43 null price values. Calendar reindexing introduces additional non-trading-day gaps. Use only past forward-fill with missingness flags, never future interpolation or realized future oil.
- Holiday table has 350 rows on 312 dates, including 12 transferred entries. The 38 excess date keys are not identical duplicates: several event/locality records can share a date. Expand applicability by store locality and aggregate flags before joining to sales. Do not multiply demand rows.
- Item/store/oil/store-transaction join keys are unique. Store transaction counts cover 1,682 dates and are realized activity, so only lagged values may be used. No actual price, customer identity, inventory or supplier data is supplied.

## UCI Online Retail

- 135,080 missing CustomerID values (24.93%) and 1,454 missing descriptions. The [UCI page](https://archive.ics.uci.edu/dataset/352/online+retail) labels missingness absent, but the downloaded workbook contains these gaps. Keep positive merchandise purchases with missing customer IDs for product forecasting; customer activity must include an unknown-ID indicator. Do not infer identities.
- 5,268 identical excess rows exist. They may be duplicate ingestion or repeated identical invoice lines. Preserve raw data and report a predeclared retain-versus-deduplicate sensitivity; the primary proposed deduplication rule retains the first complete identical line and logs affected quantity.
- 9,288 rows have cancellation-prefixed invoices; 10,624 rows have negative quantity. All cancellation rows have negative quantity; **1,336 negative-quantity rows lack the cancellation prefix**. Separate cancelled/returned/adjustment lines rather than testing InvoiceNo alone. Keep a negative-flow ledger; do not add its absolute values to purchase demand.
- 2,515 rows have zero price and two have negative price. These require explicit treatment of free items and adjustments. The proposed paid-merchandise primary target excludes nonpositive prices; a free-merchandise sensitivity is separate. Do not use future realized UnitPrice as an exogenous predictor.
- Quantity ranges from −80,995 to 80,995; preserve raw extremes and examine correction/return links without using future cancellations to rewrite earlier information availability. The dataset cannot establish whether all extreme sales represent genuine future demand.
- 3,385 rows have StockCodes outside a preliminary five-digit-plus-optional-letter pattern. This is a review flag, **not an authoritative merchandise classifier**. Freeze an explicit non-merchandise code map after checking descriptions; regex alone must not decide the final product cohort. Descriptions vary for 650 codes; use stable StockCode identifiers and training-only metadata resolution.
- There are 69 calendar dates with no records. Absence does not prove zero unconstrained demand. The final date ends at 12:50; exclude that partial-day candidate from the primary daily benchmark with a ledger. Do not fabricate remaining hours.
- An audit-only sequential scenario leaves 522,121 lines after identical-line removal, cancellation removal, remaining nonpositive quantities, nonpositive prices and provisional code-pattern exclusion. **This is a diagnostic count, not an approved final cleaned dataset.** Its full stepwise counts are in the UCI JSON. The provisional panel has 794,550 absent product/day cells within first/last positive observations; again, these are not established zero-demand days.

For every dataset, a later preprocessing ledger must report rows, units and monetary sums before/after each rule, overlap between exclusion reasons and all join cardinality checks. Numerical thresholds, encodings and imputations must be learned from training only. Audit-time inspection has not produced model-ready panels or altered raw files.

# E. Forecasting formulation

This is a proposed protocol, not a trained experiment. All targets measure recorded purchases/sales; none of these files establishes unconstrained customer demand in the presence of stockouts. No numerical accuracy, learned weights, or inventory outcomes are available yet.

| Dataset | Forecasting entity | Frequency | Target | Horizon | Rationale |
|---|---|---|---|---|---|
| Dunnhumby | PRODUCT_ID across the observed household panel; preserve STORE_ID in source and lagged store-composition summaries | Daily relative DAY | Sum of positive QUANTITY for the same product | 14 days | Pooling the sampled households reduces store–SKU sparsity while keeping distinct products. It forecasts panel purchases, not total retailer demand. Relative DAY supports seven-day periodicity but does not establish actual weekdays or holidays. |
| Favorita | store_nbr × item_nbr | Daily calendar date | max(unit_sales, 0), with negative net-sale values retained in an adjustment ledger | 16 days | Preserve original store/item structure and the competition horizon. This is a nonnegative net-sales proxy; gross purchases cannot be recovered from net daily values. |
| UCI | StockCode across destination countries | Daily calendar date | Positive, non-cancelled merchandise Quantity | 14 days | Product-level replenishment demand proxy; destination countries are not stores. Pooling countries avoids excessive sparse product-country panels. |

Keep all eligible entities as the intended study population. Determine eligibility using only information available at the earliest tuning origin: at least 180 calendar/relative days since first observed positive purchase and at least 28 positive-demand days. For UCI also require at least 100 positive merchandise invoice lines before that origin; for Dunnhumby require 100 positive lines. Favorita records are already daily aggregates, so do not reinterpret 100 daily rows as 100 transactions. Record every excluded entity and reason. These criteria define a sufficiently observed cohort, not all retail demand. Report raw, eligible and cold-start populations separately. Do not retrospectively select entities that survive or sell during final test.

Use the same entity cohort and scoring mask for all six models. If all eligible Favorita entities exceed resources, provision appropriate hardware before the confirmatory run. A smaller deterministic, training-only, stratified cohort may be a separately labelled feasibility experiment; it cannot silently replace the full study or support full-population claims. Do not change to family totals after inspecting test scores. Store×family aggregates can support descriptive audits, but do not mix item-level and family-level accuracy or assign item perishability weights to mixed families.

Daily grids start from the first observed historical purchase, without using a future last-sale date to decide availability. Set absent sales to zero only under an explicit assumption of complete transaction capture while the entity remains in the cohort. Retain flags for observed record, assumed zero, unknown availability, and global recording gap. Exclude globally unobserved days from the primary scoring mask until closure versus missing capture is established; retain them on the time axis with a mask. Report a predeclared alternative zero-fill sensitivity. Never interpolate future targets backward. The audit's first/last-observation gap counts are descriptive only and must not define the model's active window.

For UCI, reserve 2011-12-09 outside the primary daily benchmark because the extract ends at 12:50 and full-day completeness is unverified. Retain all its rows in raw data and an explicit partial-day ledger. For Dunnhumby, no calendar year, month, quarter or national holiday will be invented. Seven-day cyclic positions are relative positions, not named weekdays.

Store model-ready data only after the audit/design stage, with keys, date or relative DAY, demand, availability mask, category metadata and permitted historical covariates. Partition large files by date/store; materialize windows lazily. Preserve item units; separate fuel/weight-based quantities from packaged-unit interpretations in regime and inventory reporting.

# F. Train/validation/test methodology

Use expanding-window rolling origins, never shuffled records. The proposed boundaries are in `experiments/split_manifest.json`; no models or transformed training datasets have been built.

1. Reserve the final **four non-overlapping horizons** as the final test block: 56 days for Dunnhumby/UCI, 64 for Favorita. Favorita's Kaggle `test.csv` has no labels and is not the research test set. The research holdout comes from the end of labelled `train.csv`.
2. Immediately preceding the test, reserve four horizons for Hybrid calibration. Immediately preceding calibration, reserve four horizons for baseline hyperparameter tuning. All earlier history is the initial training window. Entity eligibility is fixed at the first tuning origin.
3. Each tuning origin fits transformations and each candidate only on data through that origin. Every supervised training window must have its entire label horizon within that training interval; purge windows crossing the cutoff. Early stopping uses an inner chronological tail of the training window, not the outer scoring horizon.
4. Select each baseline's configuration from the four tuning folds. Refit those fixed configurations at each calibration origin and save genuinely out-of-sample component forecasts. No test observations enter selection, scaling, imputation, encodings, regime thresholds or calibration.
5. Fit the final Hybrid weights on the calibration predictions. Assess calibration-period ensemble behavior prequentially: weights for calibration fold k use folds strictly before k. Fold 1 has no learned-weight assessment; an equal-weight initialization is labelled as initialization. Full-calibration fitted error is descriptive fitting error, not independent validation performance.
6. Freeze preprocessing choices, features, entity cohort, hyperparameters, epoch counts, weights, evaluation masks, inventory assumptions and analysis scripts. Save checksums and a dated freeze record before accessing test outcomes for model evaluation.
7. Refit selected components on all pre-test data with frozen configurations. For neural networks, use an epoch budget selected on development data; never early-stop on final test. Generate four sequential test-origin forecasts. Model parameters and Hybrid weights stay fixed throughout this block; lag/context inputs may incorporate observations from earlier test horizons only once they would actually be known. No within-horizon actual sales may enter a multi-step forecast. This is operational rolling evaluation, not a claim that all 56/64 days were predicted from the first origin.
8. Log any failure, retry and fallback. Never relabel Naive fallback forecasts as a successful SARIMA/LSTM/etc. Report both success coverage and paired results on a common successful cohort; also report the operational failure rate. If a component fails, the intended full Hybrid is unavailable for that case unless a fallback policy was frozen in advance.

Raw structural and data-quality summaries include the full source files as requested. That limited inspection is not a claim of a physically unread holdout. No final-test forecast performance, target-based feature choice or parameter selection has been inspected. Keep this disclosure in the paper.


Proposed date boundaries (each stage has four H-day folds):

| Dataset | Initial training through | Tuning interval | Calibration interval | Final test interval |
|---|---|---|---|---|
| dunnhumby | 543 | 544 – 599 | 600 – 655 | 656 – 711 |
| favorita | 2017-02-04 | 2017-02-05 – 2017-04-09 | 2017-04-10 – 2017-06-12 | 2017-06-13 – 2017-08-15 |
| uci | 2011-06-23 | 2011-06-24 – 2011-08-18 | 2011-08-19 – 2011-10-13 | 2011-10-14 – 2011-12-08 |

# G. Feature engineering plan

| Feature group | Dunnhumby | Favorita | UCI | Availability rule |
|---|---|---|---|---|
| Calendar | Relative DAY, relative seven-day phase, supplied WEEK_NO after verifying mapping | Day, ISO week, month, quarter, year, weekday, weekend | Same calendar features | Deterministic future dates are known; Dunnhumby calendar anchor remains unknown. |
| Lags | 1, 2, 3, 7, 14, 28, 56 | Same | Same | Past target only; no future actuals in multi-step prediction. |
| Rolling summaries | Mean, median, SD, min, max over 7, 14, 28, 56 days | Same | Same | Shift before rolling; compute at forecast origin. Include observed/masked-day counts. |
| Seasonality | Seven-day sine/cosine | Seven-day and annual calendar sine/cosine | Seven-day phase; annual calendar indicators interpreted cautiously with one year | Periods are predeclared; any empirical seasonality diagnosis uses training only. |
| Metadata | Department, commodity, brand, size; lagged store mix | Item family/class/perishable and store city/state/type/cluster | StockCode; no reliable supplied taxonomy | Fit category vocabulary on training only, with unknown category token. |
| Business | Lagged realized unit revenue, discounts, active household and basket counts, lagged promotion coverage | Lagged onpromotion, lagged store transaction count and oil; holiday/locality indicators | Lagged positive-price summaries, invoice/customer activity and country mix | Contemporaneous realized prices, transactions, redemptions and customer counts are unavailable future information. |

Future promotions are an optional known-schedule setting only if their availability at the forecast origin can be supported. The conservative primary setting uses lagged promotions. The competition supplies future promotional indicators, but that alone does not prove historical schedule lead time at every backtest origin. Do not use realized future oil prices; lag/forward-fill from information already available and retain missingness flags. Do not backward-fill early oil gaps. Future holidays require a sourced, versioned calendar; Favorita holidays must match national/regional/local scope and transferred dates. No external holiday or economic dataset has been added during this audit.

For XGBoost, use origin features plus horizon index and known future calendar features in a direct pooled multi-horizon model. For LSTM and Transformer, inputs contain past context and masks; a direct H-output head receives only allowed future calendar/schedule features. No decoder teacher forcing with actual validation/test targets. Fit scalers, numerical imputers and any target transform per fold. Use training-only price summaries with an unknown-price indicator. Preserve raw extreme quantities; flag likely adjustments using semantic rules. Any numerical clipping threshold is fitted on training and tested as a declared development sensitivity, not set from holdout extremes.

Regime tags use training data only: zero proportion/inter-demand interval for intermittent demand, positive-demand variation for volatility, within-dataset volume quantiles for low volume, training autocorrelation for seasonality, and observed promotion metadata for promotional regimes. Tags may overlap; publish their formulas and cutoffs before final evaluation. Do not call an entity stable merely because it is not intermittent. Analyze errors by regime without changing the selected model after seeing test outcomes.

# H. Five-model baseline plan

Exactly five individual baselines are permitted. All use the same target, origins and cohort.

| Baseline | Implementation and proposed bounded development search | Application to the three datasets |
|---|---|---|
| Naive | Seven-day seasonal repetition; for horizon beyond seven, repeat the last observed seven-day pattern. Predeclare last-observation fallback for insufficient valid context and log its use. No tuning. | Per entity; relative seven-day phase for Dunnhumby. |
| SARIMA | Univariate seasonal ARIMA with m=7. Candidate p,q in 0–2, d in 0–1, P,Q in 0–1, D in 0–1. Screen orders using training AICc, then evaluate at most 12 shortlisted configurations on tuning folds; cap fitting time and log convergence failures. No exogenous extension under a different model label. | Local model per product or store/item. Batch and bound concurrency; full Favorita fitting may require a larger host. |
| XGBoost | Direct pooled horizon formulation; at most 12 fixed candidate combinations of depth 3/6/8, learning rate 0.03/0.1, minimum child weight 1/10, subsample and feature fraction 0.8/1.0. Up to 1,000 trees with inner chronological early stopping. Publish exact trial list before running. | Global model separately per dataset; product/store/category identifiers encoded without target leakage. |
| LSTM | Direct H-output sequence model; at most six predeclared candidates drawn from lookback 28/56/84, hidden units 32/64, 1/2 layers, dropout 0/0.2, learning rate 0.001/0.0003, batch 64/128, maximum 50 epochs with inner-tail stopping. | Global panel model per dataset, training-only scaling and entity embeddings/unknown token; lazy windows avoid copying the full panel. |
| Transformer | Standard encoder with positional encoding and direct H-output head. At most six candidates drawn from context 28/56/84, embedding 32/64, heads 2/4, layers 1/2, dropout 0.1/0.2, learning rate 0.001/0.0003, batch 32/64 and maximum 50 epochs. d_model must be divisible by heads. | Separate global panel model for each dataset, identical information availability to the LSTM. |

These are design ranges, not tuned settings. Do not exhaustively cross every neural option. Select precise candidate lists and resource caps before training, log all trials and use the same validation metric for selection within a dataset. Three fixed neural seeds are planned to quantify initialization variability; do not choose the best seed. Nonnegative prediction clipping applies consistently and is frozen before evaluation. Report negative pre-clipping rates. No extra individual forecasting architectures are included.

# I. Hybrid architecture

The proposed primary Hybrid is a static, validation-error-weighted combination of the five selected components. It is intentionally simple enough to attribute performance and audit leakage.

For each dataset independently, compute component calibration MAE E_m over the same entity/date/horizon cells. Define a numerical safeguard eps = 1e-8 × max(1, mean absolute pre-test demand). Set a_m = 1 / max(E_m, eps) and w_m = a_m / sum_j a_j. If several E_m are numerically zero, split weight equally among those components and set others to zero. Forecast yhat_H = sum_m w_m yhat_m. Store the errors, masks, weights, calibration origin IDs, eps and configuration checksum. Weights are nonnegative and sum to one, but no superiority is implied: correlated errors can limit ensemble benefit.

Weights use only calibration forecasts generated after baseline configurations are frozen. Fit one vector per dataset, not one shared vector pooled across incompatible units. Do not fit weights on test predictions or reweight during final test. Compute the strongest individual comparator from final measured errors for reporting, while clearly distinguishing it from the development-selected comparator.

Optional adaptive extension is deferred, not implemented: use training-defined demand regimes and calibration-only regime weights with minimum calibration support and shrinkage to the static weights. Freeze its rules before test. It is a Hybrid variant in a separate analysis, not a seventh individual baseline. Begin with the static Hybrid to avoid additional selection degrees of freedom.

Ablations reuse the identical saved out-of-sample component predictions and refit subset weights only on calibration: Naive+SARIMA, SARIMA+XGBoost, XGBoost+LSTM, LSTM+Transformer, and the full Hybrid, alongside the five individuals. Add leave-one-component-out variants using only these same five components if contribution attribution is needed; a large ensemble weight alone is not evidence of marginal contribution. Reusing forecasts saves compute; include subset-weight fitting overhead separately. Never normalize full-model test-fitted weights or choose a winning subset on the test set.

# J. Evaluation metrics

Let e = prediction − observed demand and N be the number of valid scored cells. Report unit-based measures within each dataset and entity formulation:

| Metric | Definition / rule |
|---|---|
| MAE | mean(abs(e)) |
| RMSE | sqrt(mean(e²)) |
| MAPE | 100 × mean(abs(e)/y) on y>0 only; publish zero count and retained coverage; unavailable if no positive actuals. It is not an all-demand accuracy percentage. |
| sMAPE | mean(200 × abs(e)/(abs(y)+abs(prediction))); define both-zero term as zero and disclose its frequency. |
| WAPE | 100 × sum(abs(e))/sum(abs(y)); unavailable when denominator is zero. |
| WMAE | sum(w × abs(e))/sum(w). Favorita uses item perishability weights 1.25/1.00. Other datasets: N/A unless a defensible weight is predeclared; do not invent holiday weights. |
| Forecast bias | mean(e), positive means overforecast; also report 100 × sum(e)/sum(y), with zero-denominator guard. |
| Favorita NWRMSLE | sqrt(sum(w × (log1p(prediction)−log1p(y))²)/sum(w)), with nonnegative targets/predictions and item weights. |

Favorita's official competition metric is NWRMSLE, with perishable items weighted 1.25 and other items 1.00. The newly constructed historical holdout is not a Kaggle leaderboard reproduction. [Official competition evaluation](https://www.kaggle.com/c/favorita-grocery-sales-forecasting).

Primary development objective: pooled MAE (equivalent to ordering by WAPE on identical cells) for Dunnhumby/UCI, NWRMSLE for Favorita baseline tuning. The primary static Hybrid's inverse-MAE rule is predeclared rather than claimed optimal for NWRMSLE; report both objectives honestly. Report pooled metrics, macro entity summaries, per-horizon errors, per-origin errors and coverage. Use the six-row model list in the main accuracy table. Ablations/optional Hybrid variants belong in separate tables.

Hybrid improvement over each baseline is baseline error minus Hybrid error, and 100 × that difference / baseline error. If the baseline error is zero, percentage improvement is undefined; report the absolute difference. Include negative improvements. Compare generalization using within-dataset rankings and error ratios to Naive/strongest individual, not raw MAE across datasets. Three dataset case studies do not prove universal generalization.

For uncertainty, use paired origin-level loss differences with time blocks and keep all entities from a date block together to preserve shared shocks. Report origin mean, SD, sample count and intervals with the resampling unit stated. A paired block bootstrap is exploratory with only four test origins. Do not manufacture precise significance from millions of dependent rows. Use a Diebold–Mariano test with horizon-aware dependence correction only if a separately predeclared evaluation provides sufficiently many origins and diagnostics support it; otherwise p-values are N/A. Apply Holm correction across the five Hybrid-vs-individual hypotheses when tests are defensible. Tuning-fold scores are selection diagnostics, not unbiased evidence of superiority. Prequential calibration scores and held-out scores must be labelled separately.

# K. Computational evaluation plan

Measure on the same machine where feasible and record OS, CPU model, RAM, accelerator, library versions, thread counts, precision and seeds. Current host details are in `data/profiles/hardware.json`. Its 8 GiB RAM and limited free storage constrain the full Favorita experiment; successful streaming audit does not establish training feasibility.

For each dataset/model report candidate-search time, selected-model training time, inference time, total wall time, peak resident RAM, device memory where measurable, trainable parameter count, artifact bytes and latency. Neural parameter counts, SARIMA coefficient counts and XGBoost tree/node counts are structurally different; label them rather than pretending all are comparable trainable weights. Naive has no learned coefficients but does retain state. Hybrid size includes all five saved components plus weights; Hybrid training cost includes all component fitting and calibration. Report marginal combining cost separately so it cannot appear cheaper by excluding its components.

Time fitting after data preparation, excluding download, decompression, installation, visualization and reporting. State whether device transfer is included. Synchronize accelerator operations before timing. Run warm-up inference separately, then repeated batches on an identical number of entity/horizon forecasts; report median and p95 plus batch size. RAM measurement must include child workers; use sampling and acknowledge missed short peaks. On Apple hardware, unified memory is not an NVIDIA GPU-memory measure; report unsupported device measurements as unavailable.

Planning estimates, not benchmarks: UCI/Dunnhumby analysis can stream on 8 GiB, but global training should have 16–32 GiB RAM and adequate SSD space. Full item-level Favorita is better budgeted at 32–64 GiB RAM, roughly 30–60 GiB free scratch/artifact storage and, for neural trials, an optional 8–16 GiB accelerator. Exact needs depend on cohort and features. A dense float32 array alone costs 4 × entities × days × features bytes, before masks, copies, windows and optimizer state; never eagerly duplicate every lookback window.

Use a development-only pilot after authorization for the modeling stage: 100 predeclared entities, two configurations and one origin for resource measurement, not final accuracy claims. Extrapolate local SARIMA as entities × candidates × origins × measured fit time / effective workers, plus overhead. For illustration only, 200,000 entities × 12 candidates × 4 tuning origins at 0.1–1 second per fit means about 267–2,667 serial hours. This is arithmetic, not a measured runtime. Global neural/XGBoost times require an actual pilot; no honest fixed-hour promise can be made now. Freeze timeouts and feasibility decisions before test evaluation.

# L. Inventory evaluation plan

No actual inventory, procurement lead times, purchase costs, holding costs or lost-sales penalties have been verified in these archives. Inventory evaluation will therefore be an explicit scenario simulation using observed sales as a demand proxy, never as observed operational outcomes.

Proposed common baseline assumptions: daily review R=1 day; deterministic lead time L=7 days; nominal service target 95%; no backorders (lost sales), unlimited supplier capacity, no minimum order, no spoilage in the initial scenario; beginning inventory equal to training-only mean demand × (L+R) plus common safety stock. Orders arrive at the start of their arrival day, demand is served next, and replenishment orders are placed after demand from inventory position. Document day-zero pipeline and a pre-test warm-up period identically for every model.

At each review, target S_t = sum of model forecasts for the next L+R days + safety stock. Order max(0, S_t − inventory position), rounded up only for discrete-unit products. Inventory position is on-hand plus outstanding orders. Daily reviews require an additional daily inference pass from the same frozen models and weights; the four non-overlapping accuracy origins alone do not supply forecasts for every daily review. Update lag inputs only with already-observed history, charge this inference cost separately, and never tune on the daily inventory outcomes. Initial common safety stock is max(0, 1.645 × training daily demand SD × sqrt(L+R)); this assumes independent daily errors/demand for the approximation and does not guarantee achieved 95% service. A later calibrated residual-based buffer must use validation only and report its sample support.

Predeclare abstract cost units: holding 0.01 per unit per day, lost-sale penalty 1.00 per unit and ordering cost 1.00 per nonzero order. These are assumptions, not currency estimates from the retailers. Sensitivity grid: lead times 1/3/7 days, nominal service targets 90/95/99%, holding 0.005/0.01/0.02 and penalty 0.5/1/2, with the same scenarios across all six models. Do not choose the scenario that makes the Hybrid win. Unit comparability across different products is limited; report per-entity/within-dataset relative outcomes. Dunnhumby inventory is hypothetical fulfillment of panel purchases, not actual store inventory.

Measure cycle service level (cycles without lost units / evaluated cycles), fill rate (served units / requested units), positive-demand-day stockout rate, average daily on-hand, period turnover (served units / mean stock; label period and guard zero stock), holding cost, lost-sales cost, ordering cost and total cost. Account for open orders and ending inventory at the terminal boundary and report a sensitivity to salvage value. Do not call 1−stockout-day rate cycle service level. Retain per-day inventory paths and lost-unit flags. Models can improve forecast metrics yet worsen inventory costs; report that outcome if it occurs.

# M. MARIOD comparison plan

MARIOD jointly couples Transformer forecasting and hierarchical reinforcement-learning inventory control. Its public description uses daily series, a 70/15/15 split, products with at least 100 sales records, sales zero-filling and inventory forward-filling. It reports augmented inventory data and eight 80-GB A100 GPUs. Exact processed files, numeric horizon, split dates and sufficient inventory parameters are not identified for reproduction. Dataset-specific result breakdowns are unclear. [MARIOD, Sections 4–5 and data-availability statement](https://doi.org/10.3390/s25082428).

Compare methodology and provenance, not unmatched numerical scores. A reproduction would require the authors' processed data, cohort, date mapping, sensor/inventory augmentations, forecast origins and operational parameterization. The paper's Favorita reference is abbreviated and does not resolve its dataset version. The supplied original item-level competition is documented as the selected public version; no identity with the paper's unpublished processed inputs is claimed.

Do not implement a MARIOD-inspired extra forecasting baseline: the user's strict five-individual-model restriction takes precedence over that optional suggestion. The proposed Hybrid's contribution to investigate is reproducible heterogeneous forecast combination, regime-specific component behavior and its inventory implications under common assumptions. Novelty and superiority remain hypotheses. No author contact has been sent and no published MARIOD metric has been entered into the new-results tables.

# N. Complete experiment roadmap

| Stage | Work and saved artifacts | Completion gate |
|---|---|---|
| 1. Acquisition and audit — this deliverable | Preserve raw archives, hashes, source URLs, file schemas, quality counts, dataset/version differences, A–N report | All three raw datasets audited; unresolved semantics and provenance gaps disclosed. No training. |
| 2. Preprocessing | Versioned rules and row-count ledgers for duplicates, non-merchandise codes, returns, units, dates and missingness; validated joins | Reconcile input/output rows and quantities; never inflate demand through many-to-many joins. |
| 3. Forecast panels | Entity/date demand panels, observation masks, exclusion/cohort ledger and split manifest | Cohort determined from initial training only; identical targets/masks across models. |
| 4. Feature pipeline | Availability registry, lag/rolling definitions, per-fold transforms and lazy sequences | Perturb future records and verify earlier features do not change; assert max input timestamp ≤ origin and all train labels ≤ train cutoff. |
| 5. Resource pilot | Development-only fit/inference/RAM measurements and revised compute budget | Full-population feasibility or explicitly labelled limited feasibility study decided without test scores. |
| 6. Five baselines | Candidate configurations, seeds, rolling tuning predictions, losses and failure logs | Only Naive, SARIMA, XGBoost, LSTM and Transformer; all scheduled trials reported. |
| 7. Hybrid calibration | Fixed-component calibration predictions, static weights and prequential assessment | Weight provenance and nonnegativity/sum verified; no test access for fitting. |
| 8. Freeze | Checksums for cohort, preprocessing, features, configurations, weights, metrics and inventory assumptions | Dated protocol frozen before final forecast evaluation. |
| 9. Final evaluation | Refit pre-test components, four sequential frozen-model test forecasts, six-model metrics and improvements | No test tuning, test weighting or silent failure substitution. |
| 10. Contribution and generalization | Planned subset ablations, training-defined regimes, normalized comparisons and uncertainty | Negative results retained; limits from few origins and three datasets explicit. |
| 11. Inventory scenarios | Identical-policy daily paths, service, fill, stockout and assumed cost tables | Observed sales and simulated inventories/costs clearly distinguished. |
| 12. Research reporting | Five requested result tables, figures, reproducibility manifest and limitations | Answer all 15 research questions only to the extent supported by measured evidence. |

Planned final tables: dataset profile; six-model accuracy (plus bias and Favorita NWRMSLE); Hybrid improvement over each baseline; computational performance; and scenario inventory performance. Planned figures: actual-versus-predicted for each of the six models, paired error comparison, static weights, optional adaptive trajectories only if implemented, compute trade-offs, inventory paths/stockout intervals/cost composition, and relative Hybrid performance across datasets. Select example entities by a training-only documented rule and show uncertainty/coverage; do not hand-pick winning examples.

Keep raw files in `data/raw`, future panels in `data/processed`, audit JSON in `data/profiles`, code in the requested `src` subdirectories, configurations in `experiments`, and future trained artifacts, predictions, results, figures and logs in their respective directories. `models` and `predictions` remain empty at this stage. Pin package versions and seeds and maintain source checksums. The raw supplied files and public mirror have been preserved; no synthetic dataset, learned forecasting model or fabricated result was created.
