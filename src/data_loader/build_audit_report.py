"""Assemble measured profiles and proposed design; never compute forecasts."""
from pathlib import Path
import datetime as dt
import hashlib,json,zipfile,csv
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'data/profiles'
d=json.loads((P/'dunnhumby_profile.json').read_text())
f=json.loads((P/'favorita_profile.json').read_text())
u=json.loads((P/'uci_profile.json').read_text())

sources={
 'dunnhumby':{'official':'https://www.dunnhumby.com/source-files/',
 'download':'https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/download/v1.0-data/dunnhumby-complete-journey-data.zip',
 'release':'https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data',
 'status':'Public third-party mirror; hash matches release asset digest. Publisher form not submitted; no official-byte equivalence established.',
 'exact_mariod_version_verified':False},
 'favorita':{'official':'https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data',
 'local_supplied':'/Users/rajatgarg/Downloads/favorita-grocery-sales-forecasting.zip',
 'status':'User-supplied original item-level competition archive; unauthenticated API returned HTTP 401; no authentication or rules bypass.',
 'exact_mariod_version_verified':False},
 'uci':{'official':'https://archive.ics.uci.edu/dataset/352/online+retail',
 'download':'https://archive.ics.uci.edu/static/public/352/online+retail.zip',
 'status':'Downloaded directly from UCI; CC BY 4.0. Original Online Retail, not Online Retail II.',
 'exact_mariod_processed_version_verified':False},
 'mariod':{'doi':'https://doi.org/10.3390/s25082428',
 'download':'https://mdpi-res.com/d_attachment/sensors/sensors-25-02428/article_deploy/sensors-25-02428.pdf',
 'sections':'4.1–4.3; Table 1; references 63–65; data availability'}}
(P/'source_manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
manifest=[]
for key in ['dunnhumby','favorita','uci']:
 for path in sorted((ROOT/'data/raw'/key).iterdir()):
  if path.is_file():
   h=hashlib.sha256()
   with path.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
   manifest.append({'dataset':key,'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
(P/'raw_file_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

split={'status':'proposed_not_executed','models':['Naive','SARIMA','XGBoost','LSTM','Transformer','Hybrid'],
       'training_started':False,'datasets':{}}
for name,end,h in [('dunnhumby',711,14),('favorita',dt.date(2017,8,15),16),('uci',dt.date(2011,12,8),14)]:
 def shift(x,n):return x+n if isinstance(x,int) else x+dt.timedelta(days=n)
 groups={}
 for label,offset in [('tuning',12),('calibration',8),('final_test',4)]:
  a=shift(end,-offset*h+1)
  groups[label]=[{'origin':str(shift(a,k*h-1)),'forecast_start':str(shift(a,k*h)),
       'forecast_end':str(shift(a,(k+1)*h-1))} for k in range(4)]
 split['datasets'][name]={'horizon_days':h,'initial_training_end':str(shift(end,-12*h)),
   'time_axis':'relative_DAY' if name=='dunnhumby' else 'calendar_date',
   'folds':groups,'test_parameter_updates':False,'past_test_context_updates':'Only after observations become available at later origins; no weight changes.'}
(ROOT/'experiments/split_manifest.json').write_text(json.dumps(split,indent=2)+'\n')

def n(x):return f'{x:,}'
summary=[['Dunnhumby',d['rows'],d['columns'],'DAY 1–711 (calendar anchor unknown)',d['stores'],d['products'],d['customers'],'Positive product quantity','daily relative DAY'],
 ['Favorita',f['rows'],f['columns'],f['date_min']+' to '+f['date_max'],f['stores'],f['products'],'not supplied','Nonnegative daily unit_sales proxy','daily'],
 ['UCI Online Retail',u['rows'],u['columns'],'2010-12-01 to 2011-12-09','online channel',u['products'],u['customers'],'Positive non-cancelled merchandise quantity','daily aggregation']]
with (ROOT/'results/dataset_profile_table.csv').open('w') as out:
 writer=csv.writer(out);writer.writerow(['Dataset','Rows','Columns','Date Range','Stores','Products','Customers','Target','Frequency']);writer.writerows(summary)
table='| Dataset | Raw rows | Columns | Coverage | Stores | Observed products | Customers |\n|---|---:|---:|---|---:|---:|---:|\n'
for row in summary:
 table+='| '+' | '.join(n(x) if isinstance(x,int) else str(x) for x in row[:7])+' |\n'
report=f'''# A. Dataset acquisition status

Audit completed on {dt.date.today().isoformat()}. All three requested datasets have been accessed and fully scanned for the reported raw-data statistics. **No forecasting model has been trained.** The experimental design below is proposed; there are no accuracy results, learned Hybrid weights or simulated inventory results.

| Dataset | Acquisition outcome | Version and provenance limitation |
|---|---|---|
| Dunnhumby – The Complete Journey | Public full CSV archive obtained from a [GitHub release](https://github.com/bltap-plmarket/dunnhumby-complete-journey/releases/tag/v1.0-data). | [Publisher](https://www.dunnhumby.com/source-files/) requires a registration/terms form. It was not submitted. Mirror checksum matches its release digest; identity with publisher bytes or MARIOD's processed version is unverified. |
| Corporación Favorita Grocery Sales | Your supplied ZIP was copied into the project and its original item-level files audited. | [Official Kaggle source](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data). Direct unauthenticated API access returned 401; your archive resolved access. This is the original competition, not the later family-level Store Sales dataset. |
| UCI Online Retail | Downloaded the original XLSX directly from [UCI](https://archive.ics.uci.edu/dataset/352/online+retail). | The date/row coverage matches the named source, but identity with MARIOD's cleaned input is not established. Source license: CC BY 4.0. |

Original archives are preserved in `data/raw`. SHA-256 hashes and sizes for every retained raw file are in `data/profiles/raw_file_manifest.json`; acquisition details are in `source_manifest.json`. Favorita train expands to **4,997,452,288 bytes**. Low disk space interrupted an initial extraction; the incomplete generated CSV was removed and the entire train file was subsequently streamed successfully from its 7z archive. Your original Downloads ZIP and the project archive copy are preserved. No additional retail dataset was used.

# B. Dataset profiles

{table}

Rows count transaction **lines** for Dunnhumby/UCI and daily store/item records for Favorita. They are not comparable counts of shopping trips. Product counts above come from observed sales records, not the product master. Missing customer IDs are not counted as an extra customer. Raw counts precede any proposed filtering.

| Profile detail | Dunnhumby | Favorita | UCI |
|---|---|---|---|
| Main file size | 141,742,346 uncompressed bytes | 4,997,452,288 uncompressed bytes; 474,092,593-byte train 7z | 23,715,344-byte XLSX |
| Archive size | {n(d['archive']['bytes'])} bytes | {n(f['archive']['bytes'])} bytes | Preserved ZIP; exact bytes in manifest |
| Time representation | DAY 1–711; WEEK_NO 1–102 | 1,688 calendar days, 1,684 with train records | 374 calendar days, 305 with records; timestamp endpoints 08:26 and 12:50 |
| Trips / invoices | {n(d['baskets'])} baskets | No transaction IDs per sales row; store-level counts in transactions.csv | {n(u['invoices'])} invoice identifiers |
| Catalog / categories | 92,353 product-master rows; 44 department labels, 308 commodities, 2,383 subcommodities including blank labels | 4,100 item-master rows; 33 families, 337 classes | No category master; 650 StockCodes have multiple descriptions |
| Customer coverage | 2,500 households; demographics for 801 only | No household/customer identifiers | 4,372 known customers; 38 destination countries |
| Prices | Realized sales value and discounts; no explicit shelf-price field | No item price field | UnitPrice in GBP |
| Promotions | Causal display/mailer, campaigns, coupons, redemption tables | onpromotion plus holiday/event calendar | No explicit promotion field |
| External variables | Selected household demographics; no verified calendar anchor | Store metadata, holidays, oil, daily store transaction counts | Country; no supplied holiday/economic series |
| Actual inventory | Absent | Absent | Absent |

Machine-readable full profiles: `data/profiles/dataset_profile.json`, with separate per-dataset JSON files, schema, per-column null/distinct counts and audit methods. Supporting table sizes:

| Dataset | Table | Rows | Columns |
|---|---|---:|---:|
'''
for name,prefix,tables in [('Dunnhumby','',d['tables']),('Favorita','',f['tables'])]:
 for name2,v in tables.items():report+=f"| {name} | {name2} | {n(v['rows'])} | {v['columns']} |\n"
report+='''
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

'''
report+=f'''- Full streaming scan covered **{n(f['rows'])}** train rows, not a sample. IDs are consecutive; no duplicate full rows or duplicate store/item/date keys were found. All item and store references match the masters.
- `onpromotion` has **{n(f['missing']['onpromotion'])}** missing entries ({100*f['missing']['onpromotion']/f['rows']:.2f}% of rows). Unknown promotion is distinct from false: retain an unknown indicator; do not automatically treat the early missing block as “no promotion.” Other train columns contain no parser-null values.
- There are **{n(f['counts']['negative_unit_sales_rows'])}** negative-sales rows, **{n(f['counts']['fractional_unit_sales_rows'])}** fractional values and no explicit zero-sales rows. Negative values range down to {f['unit_sales_min']:g}; the maximum is {f['unit_sales_max']:g}. Preserve fractional units. Retain negative adjustments separately and document nonnegative clipping as a target definition, not recovery of gross demand.
- There are **{n(f['store_item_pairs'])}** observed store/item pairs. The full catalog cross-product has {n(f['absent_master_grid_days'])} absent cells; even within each pair's first/last observations there are {n(f['absent_days_within_first_last_observation'])} absent dates. These include assortment and capture ambiguity. Missing rows cannot all be declared observed zero sales.
'''
report+='''- No train records exist for Christmas Day in 2013–2016. This may reflect closure rather than missing collection, but the archive does not prove the cause. Keep these dates on the time axis and record their treatment explicitly.
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

'''
report+=(ROOT/'experiments/experimental_design.md').read_text()
# Add concrete split table inside methodology.
split_table='\nProposed date boundaries (each stage has four H-day folds):\n\n| Dataset | Initial training through | Tuning interval | Calibration interval | Final test interval |\n|---|---|---|---|---|\n'
for name,v in split['datasets'].items():
 intervals=[v['folds'][k][0]['forecast_start']+' – '+v['folds'][k][-1]['forecast_end'] for k in ['tuning','calibration','final_test']]
 split_table+='| '+name+' | '+v['initial_training_end']+' | '+' | '.join(intervals)+' |\n'
report=report.replace('# G. Feature engineering plan',split_table+'\n# G. Feature engineering plan')
(ROOT/'results/dataset_audit_and_experimental_roadmap.md').write_text(report)
print('Wrote A–N report, manifests, profile table, and proposed split manifest.')
