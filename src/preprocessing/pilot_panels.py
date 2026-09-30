"""Deterministic development-only panels. Raw files never change.

Full cohort eligibility is computed before pilot selection. Only 12 entities per
dataset are materialized; these files must not be described as full study panels.
"""
from pathlib import Path
import argparse, collections, hashlib, json, subprocess, sys, zipfile, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/pilot';OUT.mkdir(parents=True,exist_ok=True)
SPLIT=json.loads((ROOT/'experiments/split_manifest.json').read_text())['datasets']
BASE={'uci':'2010-12-01','favorita':'2013-01-01'}

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')
def tindex(dataset,value):
    return int(value) if dataset=='dunnhumby' else (pd.Timestamp(value)-pd.Timestamp(BASE[dataset])).days+1
def bounds(dataset):
    s=SPLIT[dataset]
    return tindex(dataset,s['initial_training_end']),tindex(dataset,s['folds']['calibration'][1]['forecast_end'])
def times(dataset,series):
    return series.astype(int) if dataset=='dunnhumby' else (pd.to_datetime(series)-pd.Timestamp(BASE[dataset])).dt.days+1
def select(dataset,stats,cutoff):
    """Input aggregates MUST contain only records at/before cutoff."""
    stats=stats.copy();stats['entity']=stats.entity.astype(str)
    stats['history_days']=cutoff-stats['first']+1
    stats['eligible']=(stats.history_days>=180)&(stats.positive_days>=28)
    if dataset!='favorita':stats['eligible'] &= stats.positive_lines>=100
    stats['reason']=np.where(stats.eligible,'eligible','insufficient_initial_training_history_or_activity')
    stats['training_mean']=stats.quantity/stats.history_days.clip(lower=1)
    eligible=stats[stats.eligible].copy()
    # Stable ranks break ties by entity identifier; quantiles use training only.
    eligible=eligible.sort_values(['training_mean','entity'])
    eligible['volume']=np.minimum(2,np.arange(len(eligible))*3//len(eligible))
    eligible['volume']=eligible.volume.map({0:'low',1:'medium',2:'high'})
    eligible['selection_hash']=eligible.entity.map(lambda e:hashlib.sha256((dataset+':'+e).encode()).hexdigest())
    chosen=eligible.sort_values('selection_hash').groupby('volume',sort=True).head(4).copy()
    assert len(chosen)==12,'Insufficient entities; never replace based on validation scores'
    stats=stats.merge(eligible[['entity','volume','selection_hash']],on='entity',how='left',validate='one_to_one')
    stats['pilot_selected']=stats.entity.isin(chosen.entity)
    stats['selection_cutoff']=cutoff
    stats.to_csv(OUT/f'{dataset}_cohort_eligibility.csv.gz',index=False)
    stats[~stats.eligible].to_csv(OUT/f'{dataset}_cohort_exclusions.csv.gz',index=False)
    chosen.to_csv(OUT/f'{dataset}_pilot_cohort.csv',index=False)
    return chosen,stats

def stats_from_lines(df,cutoff):
    a=df[(df.t<=cutoff)&df.demand.gt(0)]
    return a.groupby('entity').agg(first=('t','min'),positive_days=('t','nunique'),
        positive_lines=('demand','size'),quantity=('demand','sum')).reset_index()

def merge_checked(left,right,on,ledger,label):
    before=len(left); qty=float(left.demand.sum())
    assert not right.duplicated(on).any(),f'duplicate right keys {label}'
    out=left.merge(right,on=on,how='left',validate='many_to_one')
    assert len(out)==before and np.isclose(out.demand.sum(),qty)
    ledger.setdefault('joins',[]).append({'table':label,'keys':on,'rows_before':before,'rows_after':len(out),
        'quantity_before':qty,'quantity_after':float(out.demand.sum()),'cardinality':'many_to_one'})
    return out

def grid(dataset,daily,chosen,global_days,end,ledger):
    chunks=[]
    for row in chosen.itertuples():
        e=str(row.entity); days=np.arange(int(row.first),end+1)
        g=pd.DataFrame({'entity':e,'t':days})
        g=g.merge(daily[daily.entity.eq(e)],on=['entity','t'],how='left',validate='one_to_one')
        g['observation_mask']=g.demand.notna()
        g['global_recording_mask']=g.t.isin(global_days)
        g['assumed_zero']=~g.observation_mask & g.global_recording_mask
        # No zero is represented as directly observed if its underlying row is absent.
        g.loc[g.assumed_zero,'demand']=0.0
        g['availability_mask']=np.where(g.observation_mask,1,np.nan)
        g['score_mask']=g.global_recording_mask & g.demand.notna()
        g['availability_status']=np.where(g.observation_mask,'recorded_sale_or_adjustment','unknown')
        g['volume']=row.volume
        for c in ['promotion','price','transactions','customers','oil','holiday']:
            if c not in g:g[c]=np.nan
        g['weight']=g.get('weight',pd.Series(1.,index=g.index)).fillna(1.)
        for c in ['category','store']:
            if c in g:
                known=g[c].dropna()
                g[c]=g[c].fillna(known.iloc[0] if len(known) else 'unknown')
        g['date']=g.t.astype(str) if dataset=='dunnhumby' else (pd.Timestamp(BASE[dataset])+pd.to_timedelta(g.t-1,unit='D')).dt.strftime('%Y-%m-%d')
        chunks.append(g)
    panel=pd.concat(chunks,ignore_index=True).sort_values(['entity','t'])
    assert not panel.duplicated(['entity','t']).any()
    ledger.update(pilot_entity_count=len(chosen),final_model_ready_rows=len(panel),
        panel_t_min=int(panel.t.min()),panel_t_max=int(panel.t.max()),observed_panel_rows=int(panel.observation_mask.sum()),
        assumed_zero_rows=int(panel.assumed_zero.sum()),masked_global_gap_rows=int((~panel.global_recording_mask).sum()),
        unknown_availability_rows=int(panel.availability_mask.isna().sum()),panel_quantity=float(panel.demand.sum()),
        scope='12-entity deterministic development pilot; not full-population preprocessing',
        final_test_access='No final-test rows included in transformations, features or model panels')
    panel.to_csv(OUT/f'{dataset}_panel.csv.gz',index=False)
    panel[['entity','t','date','observation_mask','availability_mask','availability_status','assumed_zero','score_mask']].to_csv(OUT/f'{dataset}_masks.csv.gz',index=False)
    dump(OUT/f'{dataset}_preprocessing_ledger.json',ledger)
    return panel

def uci():
    start=time.perf_counter();cut,end=bounds('uci')
    d=pd.read_excel(ROOT/'data/raw/uci/Online Retail.xlsx',dtype={'InvoiceNo':'string','StockCode':'string','CustomerID':'string'})
    d['t']=times('uci',d.InvoiceDate.dt.normalize());d=d[d.t<=end].copy()
    ledger={'dataset':'uci','raw_original_rows_from_completed_audit':541909,'development_input_rows':len(d),
            'quantity_before':int(d.Quantity.sum()),'development_cutoff':end,'eligibility_cutoff':cut,'stages':[]}
    global_days=set(d.t.unique());d['source_row']=d.index+2
    d[d.Quantity.lt(0)].to_csv(OUT/'uci_returns_adjustments.csv.gz',index=False)
    exclude_codes={'AMAZONFEE','BANK CHARGES','C2','D','DOT','M','POST','S'}
    # Code map established from initial-training descriptions, not regex shape.
    dump(OUT/'uci_nonmerchandise_rules.json',{'codes':sorted(exclude_codes),'prefix':'GIFT_',
        'basis':'Initial-training descriptions: fees, carriage, discounts, postage, manual and sample entries, gift vouchers. Nonstandard merchandise codes retained.'})
    removed=[]
    def stage(reason,mask):
        nonlocal d
        before=len(d);q=float(d.Quantity.sum());bad=d[mask].copy();bad['exclusion_reason']=reason;removed.append(bad)
        d=d[~mask].copy();ledger['stages'].append({'rule':reason,'rows_before':before,'rows_removed':len(bad),
            'rows_retained':len(d),'quantity_before':q,'quantity_after':float(d.Quantity.sum())})
    stage('identical_line_excess',d.drop(columns=['source_row']).duplicated())
    stage('cancelled_invoice',d.InvoiceNo.str.upper().str.startswith('C').fillna(False))
    stage('remaining_nonpositive_quantity',d.Quantity.le(0))
    stage('nonpositive_price',d.UnitPrice.le(0))
    stage('explicit_nonmerchandise_code',d.StockCode.str.upper().isin(exclude_codes)|d.StockCode.str.upper().str.startswith('GIFT_'))
    pd.concat(removed).to_csv(OUT/'uci_row_exclusions.csv.gz',index=False)
    d['entity']=d.StockCode;d['demand']=d.Quantity.astype(float)
    chosen,stats=select('uci',stats_from_lines(d,cut),cut)
    ledger.update(eligible_entity_count=int(stats.eligible.sum()),excluded_entity_count=int((~stats.eligible).sum()),
        clean_development_rows=len(d),quantity_after=float(d.demand.sum()),negative_quantity_rows=int(sum(x.Quantity.lt(0).sum() for x in removed)),
        missing_customer_rows_retained=int(d.CustomerID.isna().sum()),missing_description_rows_retained=int(d.Description.isna().sum()))
    pilot=d[d.entity.isin(chosen.entity)].copy()
    daily=pilot.groupby(['entity','t']).agg(demand=('demand','sum'),price=('UnitPrice','mean'),
        transactions=('InvoiceNo','nunique'),customers=('CustomerID','nunique'),returned_units=('Quantity',lambda x:0)).reset_index()
    daily['category']='unprovided';daily['store']='online';daily['weight']=1.
    ledger.update(pilot_transaction_rows=len(pilot),pilot_quantity_before_aggregation=float(pilot.demand.sum()),
        aggregation_rows=len(daily),aggregation_quantity=float(daily.demand.sum()),runtime_seconds=time.perf_counter()-start)
    grid('uci',daily,chosen,global_days,end,ledger)

def dunnhumby():
    start=time.perf_counter();cut,end=bounds('dunnhumby')
    z=zipfile.ZipFile(ROOT/'data/raw/dunnhumby/dunnhumby-complete-journey-data.zip')
    with z.open('transaction_data.csv') as f:d=pd.read_csv(f)
    d=d[d.DAY<=end].copy();d['t']=d.DAY;global_days=set(d.t)
    ledger={'dataset':'dunnhumby','raw_original_rows_from_completed_audit':2595732,'development_input_rows':len(d),
       'quantity_before':float(d.QUANTITY.sum()),'development_cutoff':end,'eligibility_cutoff':cut}
    bad=d[d.QUANTITY.le(0)].copy();bad['exclusion_reason']='nonpositive_quantity';bad.to_csv(OUT/'dunnhumby_row_exclusions.csv.gz',index=False)
    dup=d.duplicated();assert not dup.any(),'Changed raw duplicate state'
    d=d[d.QUANTITY.gt(0)].copy();d['entity']=d.PRODUCT_ID.astype(str);d['demand']=d.QUANTITY.astype(float)
    chosen,stats=select('dunnhumby',stats_from_lines(d,cut),cut)
    with z.open('product.csv') as f:products=pd.read_csv(f)
    products['entity']=products.PRODUCT_ID.astype(str)
    products['category']=products.COMMODITY_DESC.astype('string').str.strip().replace('','unknown')
    pilot=d[d.entity.isin(chosen.entity)].copy();pilot['price']=pilot.SALES_VALUE/pilot.QUANTITY
    pilot['promotion']=pilot.RETAIL_DISC.ne(0).astype(float)
    daily=pilot.groupby(['entity','t']).agg(demand=('demand','sum'),price=('price','mean'),promotion=('promotion','mean'),
        transactions=('BASKET_ID','nunique'),customers=('household_key','nunique'),stores_active=('STORE_ID','nunique')).reset_index()
    daily=merge_checked(daily,products[['entity','category']],['entity'],ledger,'product_master')
    daily['store']='household_panel';daily['weight']=1.
    # Causal and coupons intentionally not joined to demand: no verified schedule availability.
    # Discount proxy is realized historical activity only, never a future promotion schedule.
    ledger.update(negative_quantity_rows=0,rows_removed=len(bad),rows_retained=len(d),quantity_after=float(d.demand.sum()),
        duplicate_excess_rows=0,eligible_entity_count=int(stats.eligible.sum()),excluded_entity_count=int((~stats.eligible).sum()),
        pilot_transaction_rows=len(pilot),pilot_quantity_before_aggregation=float(pilot.demand.sum()),
        aggregation_quantity=float(daily.demand.sum()),aggregation_rows=len(daily),
        missing_values='Raw numeric inputs complete. Blank product categories normalized to unknown. Demographic join omitted; all households retained.',
        promotions='Lagged realized-discount proxy only. Causal schedule/coupon/redemption features deferred due unavailable forecast-time schedule provenance; no unsafe many-to-many join.',
        runtime_seconds=time.perf_counter()-start)
    grid('dunnhumby',daily,chosen,global_days,end,ledger)

def favorita_stream(end):
    proc=subprocess.Popen(['bsdtar','-xOf',str(ROOT/'data/raw/favorita/train.csv.7z'),'train.csv'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    prev=None
    try:
        for c in pd.read_csv(proc.stdout,chunksize=1000000,dtype={'id':'int64','date':'string','store_nbr':'int16','item_nbr':'int32','unit_sales':'float64','onpromotion':'string'}):
            # Sorted source dates are checked before stopping early at the development boundary.
            assert c.date.is_monotonic_increasing and (prev is None or c.date.iloc[0]>=prev)
            prev=c.date.iloc[-1];c['t']=times('favorita',c.date)
            stop=c.t.max()>end;c=c[c.t<=end].copy()
            if len(c):
                c['entity']=c.store_nbr.astype(str)+':'+c.item_nbr.astype(str)
                yield c
            if stop:break
    finally:
        if proc.poll() is None:proc.terminate()
        proc.stdout.close();proc.wait()

def favorita():
    start=time.perf_counter();cut,end=bounds('favorita');parts=[];returns=[];global_days=set();rows=0;q=0.;neg=0;miss=0;clipped=0.
    print('Favorita eligibility pass',flush=True)
    for c in favorita_stream(end):
        rows+=len(c);q+=float(c.unit_sales.sum());global_days.update(c.t.unique());neg+=int(c.unit_sales.lt(0).sum());miss+=int(c.onpromotion.isna().sum())
        returns.append(c[c.unit_sales.lt(0)]);clipped+=float(c.unit_sales.clip(lower=0).sum())
        a=c[(c.t<=cut)&c.unit_sales.gt(0)]
        parts.append(a.groupby('entity').agg(first=('t','min'),positive_days=('t','size'),quantity=('unit_sales','sum')).reset_index())
        if rows%10000000==0:print('Favorita rows scanned',rows,flush=True)
    stats=pd.concat(parts).groupby('entity').agg(first=('first','min'),positive_days=('positive_days','sum'),quantity=('quantity','sum')).reset_index();stats['positive_lines']=stats.positive_days
    chosen,stats=select('favorita',stats,cut)
    ret=pd.concat(returns);ret.to_csv(OUT/'favorita_returns_adjustments.csv.gz',index=False)
    ledger={'dataset':'favorita','raw_original_rows_from_completed_audit':125497040,'development_input_rows':rows,
      'quantity_before':q,'quantity_after':clipped,'negative_quantity_rows':neg,'missing_onpromotion_rows':miss,
      'rows_removed':0,'rows_retained':rows,'negative_rule':'unit_sales retained separately; demand=max(unit_sales,0)',
      'fractional_sales':'preserved float64, never rounded','eligibility_cutoff':cut,'development_cutoff':end,
      'eligible_entity_count':int(stats.eligible.sum()),'excluded_entity_count':int((~stats.eligible).sum()),
      'duplicate_policy':'Completed full audit proved unique date/store/item keys; pilot keys asserted again.',
      'returns_quantity':float(ret.unit_sales.sum())}
    print('Favorita selected entity extraction pass',flush=True)
    chunks=[]
    for c in favorita_stream(end):chunks.append(c[c.entity.isin(chosen.entity)].copy())
    a=pd.concat(chunks);assert not a.duplicated(['entity','t']).any()
    a['demand']=a.unit_sales.clip(lower=0);a['promotion']=a.onpromotion.str.lower().map({'true':1.,'false':0.})
    a['returned_units']=(-a.unit_sales).clip(lower=0)
    daily=a[['entity','t','date','store_nbr','item_nbr','demand','promotion','returned_units']].copy()
    raw=ROOT/'data/raw/favorita'
    items=pd.read_csv(raw/'items.csv').rename(columns={'family':'category'});items['weight']=1+0.25*items.perishable
    stores=pd.read_csv(raw/'stores.csv');trans=pd.read_csv(raw/'transactions.csv');trans['t']=times('favorita',trans.date)
    daily=merge_checked(daily,items,['item_nbr'],ledger,'items')
    daily=merge_checked(daily,stores,['store_nbr'],ledger,'stores')
    daily=merge_checked(daily,trans[['t','store_nbr','transactions']],['t','store_nbr'],ledger,'transactions')
    oil=pd.read_csv(raw/'oil.csv');oil['t']=times('favorita',oil.date);oil=oil[oil.t<=end]
    # Forward-only imputation, before joining, on unique date keys.
    oil=oil.set_index('t').reindex(range(1,end+1));oil['oil']=oil.dcoilwtico.ffill();oil=oil.reset_index(names='t')
    daily=merge_checked(daily,oil[['t','oil']],['t'],ledger,'oil_past_forward_fill')
    holidays=pd.read_csv(raw/'holidays_events.csv');holidays['t']=times('favorita',holidays.date);holidays=holidays[holidays.t<=end]
    h=[]
    for st in stores.itertuples():
        applicable=holidays[(holidays.locale.eq('National'))|(holidays.locale.eq('Regional')&holidays.locale_name.eq(st.state))|(holidays.locale.eq('Local')&holidays.locale_name.eq(st.city))].copy()
        applicable['holiday']=(~applicable.transferred & ~applicable.type.eq('Work Day')).astype(int)
        x=applicable.groupby('t').holiday.max().reset_index();x['store_nbr']=st.store_nbr;h.append(x)
    h=pd.concat(h);daily=merge_checked(daily,h,['t','store_nbr'],ledger,'holidays_aggregated_by_applicable_locality')
    daily['holiday']=daily.holiday.fillna(0);daily['store']=daily.store_nbr.astype(str)
    ledger.update(pilot_sales_rows=len(a),pilot_quantity_before_joins=float(a.demand.sum()),
        pilot_quantity_after_joins=float(daily.demand.sum()),runtime_seconds=time.perf_counter()-start)
    panel=grid('favorita',daily.drop(columns='date'),chosen,global_days,end,ledger)
    # Static catalog weights/IDs must be carried onto zero/unknown-date rows too.
    lookup=daily.drop_duplicates('entity').set_index('entity')
    for col in ['weight','category','store','store_nbr','item_nbr','perishable','city','state','type','cluster']:
        panel[col]=panel.entity.map(lookup[col])
    # Oil, holidays, store transactions are date/store series, not conditional on item sales.
    panel=panel.drop(columns=['oil','holiday','transactions']).merge(oil[['t','oil']],on='t',how='left',validate='many_to_one')
    panel=panel.merge(trans[['t','store_nbr','transactions']],on=['t','store_nbr'],how='left',validate='many_to_one')
    panel=panel.merge(h,on=['t','store_nbr'],how='left',validate='many_to_one');panel['holiday']=panel.holiday.fillna(0)
    assert len(panel)==ledger['final_model_ready_rows'] and not panel.duplicated(['entity','t']).any()
    panel.to_csv(OUT/'favorita_panel.csv.gz',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['uci','dunnhumby','favorita']);a=ap.parse_args()
    globals()[a.dataset]();print(a.dataset,'pilot preprocessing complete',flush=True)
