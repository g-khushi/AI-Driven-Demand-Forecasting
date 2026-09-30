"""Read-only raw-data audit. No forecasting models or synthetic observations.

Run with bundled Python. Outputs contain audit statistics, not model-ready data.
Favorita train is streamed from 7z to avoid a multi-GB extracted CSV.
"""
from pathlib import Path
import argparse, collections, datetime as dt, hashlib, json, os, platform
import subprocess, sys, time, zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.audit_deps'))
import psutil
OUT = ROOT / 'data/profiles'
OUT.mkdir(parents=True, exist_ok=True)

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False)+'\n')

def metadata(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': h.hexdigest()}

def basic(df):
    return {'rows':len(df), 'columns':len(df.columns), 'schema':{c:str(t) for c,t in df.dtypes.items()},
            'missing':{c:int(v) for c,v in df.isna().sum().items()},
            'distinct_nonmissing':{c:int(v) for c,v in df.nunique().items()},
            'exact_duplicate_excess_rows':int(df.duplicated().sum())}

def gap_profile(frame, entity, date, target):
    # Descriptive audit only: absence is not established zero demand.
    g=frame.groupby([entity,date],observed=True)[target].sum().reset_index()
    span=g.groupby(entity,observed=True)[date].agg(['min','max','count'])
    possible=((span['max']-span['min']).dt.days+1).sum()
    return {'observed_entity_days':len(g), 'observed_zero_sum_entity_days':int(g[target].eq(0).sum()),
            'within_first_last_observation_possible_days':int(possible),
            'absent_entity_days_within_first_last_observation':int(possible-len(g)),
            'interpretation':'Absent records are not proven zero demand; no availability/inventory flag.'}

def uci():
    path=ROOT/'data/raw/uci/Online Retail.xlsx'
    print('UCI reading workbook',flush=True)
    df=pd.read_excel(path,dtype={'InvoiceNo':'string','StockCode':'string','CustomerID':'string'})
    p=basic(df); p['file']=metadata(path)
    dates=df.InvoiceDate; day=dates.dt.normalize()
    cancelled=df.InvoiceNo.str.upper().str.startswith('C').fillna(False)
    p.update({'dataset':'UCI Online Retail', 'status':'measured_full_file',
        'date_min':str(dates.min()),'date_max':str(dates.max()),'calendar_days':int((day.max()-day.min()).days+1),
        'days_with_records':int(day.nunique()),'stores':None,'channel':'online',
        'products':int(df.StockCode.nunique()),'customers':int(df.CustomerID.nunique()),
        'invoices':int(df.InvoiceNo.nunique()),'countries':int(df.Country.nunique()),
        'categories':None,'negative_quantity_rows':int(df.Quantity.lt(0).sum()),
        'zero_quantity_rows':int(df.Quantity.eq(0).sum()),'cancelled_invoice_rows':int(cancelled.sum()),
        'negative_quantity_without_cancellation_rows':int((df.Quantity.lt(0)&~cancelled).sum()),
        'cancellation_nonnegative_quantity_rows':int((cancelled&df.Quantity.ge(0)).sum()),
        'zero_price_rows':int(df.UnitPrice.eq(0).sum()),'negative_price_rows':int(df.UnitPrice.lt(0).sum()),
        'quantity_min':int(df.Quantity.min()),'quantity_max':int(df.Quantity.max()),
        'price_min':float(df.UnitPrice.min()),'price_max':float(df.UnitPrice.max()),
        'nonstandard_stockcode_rows':int((~df.StockCode.str.fullmatch(r'\d{5}[A-Za-z]?').fillna(False)).sum()),
        'stockcodes_multiple_descriptions':int((df.groupby('StockCode').Description.nunique()>1).sum()),
        'missing_calendar_days':[str(x.date()) for x in pd.date_range(day.min(),day.max()).difference(pd.DatetimeIndex(day.unique()))],
        'last_day_last_timestamp':str(dates.max()),
        'available_external_variables':['Country'], 'promotions':'not provided','holidays':'not provided',
        'inventory':'not provided','returns':'negative quantity / cancellation proxies; reason not always known'})
    # Counts for explicit, proposed filtering, no raw mutation and no model-ready export.
    remaining=pd.Series(True,index=df.index); stages=[]
    for name, bad in [('exact_duplicate_excess',df.duplicated()),('cancelled_invoice',cancelled),
                      ('nonpositive_quantity',df.Quantity.le(0)),('nonpositive_price',df.UnitPrice.le(0)),
                      ('nonstandard_stockcode_pending_manual_review',~df.StockCode.str.fullmatch(r'\d{5}[A-Za-z]?').fillna(False))]:
        removed=remaining&bad; remaining &= ~bad
        stages.append({'rule':name,'removed_rows_sequential':int(removed.sum()),'remaining_rows':int(remaining.sum())})
    p['proposed_filter_impact_not_applied']=stages
    d=df.loc[remaining,['StockCode','Quantity']].copy();d['date']=day[remaining]
    p['provisional_positive_merchandise_panel']=gap_profile(d,'StockCode','date','Quantity')
    p['duplicate_caveat']='Identical invoice lines may be legitimate repeated line entries. Preserve raw; report retain-vs-deduplicate sensitivity.'
    save('uci_profile.json',p);print('UCI complete',len(df),flush=True)

def dunnhumby():
    path=ROOT/'data/raw/dunnhumby/dunnhumby-complete-journey-data.zip'
    z=zipfile.ZipFile(path); manifest=[{'name':i.filename,'bytes_uncompressed':i.file_size,'bytes_compressed':i.compress_size} for i in z.infolist()]
    print('Dunnhumby archive',manifest,flush=True)
    tables={}; tx=None; product=None
    for member in z.namelist():
        if not member.endswith('.csv') or 'causal' in member.lower():continue
        print('Dunnhumby reading',member,flush=True)
        with z.open(member) as f: df=pd.read_csv(f,low_memory=False)
        tables[Path(member).name]=basic(df)
        if Path(member).name=='transaction_data.csv':tx=df
        if Path(member).name=='product.csv':product=df
    assert tx is not None and product is not None
    p={'dataset':'Dunnhumby The Complete Journey','status':'measured_public_mirror', 'archive':metadata(path),
       'archive_members':manifest,'tables':tables,'rows':len(tx),'columns':len(tx.columns),
       'day_min':int(tx.DAY.min()),'day_max':int(tx.DAY.max()),'days_with_records':int(tx.DAY.nunique()),
       'week_min':int(tx.WEEK_NO.min()),'week_max':int(tx.WEEK_NO.max()),'weeks_with_records':int(tx.WEEK_NO.nunique()),
       'calendar_date_range':None,'stores':int(tx.STORE_ID.nunique()),'products':int(tx.PRODUCT_ID.nunique()),
       'product_master_rows':len(product),'customers':int(tx.household_key.nunique()),'baskets':int(tx.BASKET_ID.nunique()),
       'departments':int(product.DEPARTMENT.nunique()),'commodities':int(product.COMMODITY_DESC.nunique()),
       'subcommodities':int(product.SUB_COMMODITY_DESC.nunique()),
       'negative_quantity_rows':int(tx.QUANTITY.lt(0).sum()),'zero_quantity_rows':int(tx.QUANTITY.eq(0).sum()),
       'quantity_min':int(tx.QUANTITY.min()),'quantity_max':int(tx.QUANTITY.max()),
       'negative_sales_value_rows':int(tx.SALES_VALUE.lt(0).sum()),'zero_sales_value_rows':int(tx.SALES_VALUE.eq(0).sum()),
       'positive_quantity_zero_sales_rows':int((tx.QUANTITY.gt(0)&tx.SALES_VALUE.eq(0)).sum()),
       'transaction_product_orphans':int((~tx.PRODUCT_ID.isin(product.PRODUCT_ID)).sum()),
       'discount_nonzero_rows':{c:int(tx[c].ne(0).sum()) for c in ['RETAIL_DISC','COUPON_DISC','COUPON_MATCH_DISC']},
       'inventory':'not provided','holidays':'not provided; calendar anchor unverified',
       'returns':'no explicit return/cancellation field',
       'unit_price':'not explicit; SALES_VALUE/QUANTITY is realized unit revenue for positive quantity only',
       'demand_scope':'Purchases by the sampled household panel, not complete store sales.'}
    q=tx.loc[tx.QUANTITY.gt(0),['PRODUCT_ID','DAY','QUANTITY']].copy()
    q['relative_date']=pd.Timestamp('2000-01-01')+pd.to_timedelta(q.DAY,unit='D')
    # This internal arbitrary timestamp computes integer gaps only, never a calendar claim.
    p['positive_product_day_panel']=gap_profile(q,'PRODUCT_ID','relative_date','QUANTITY')
    p['positive_product_day_panel']['date_note']='Gap computation uses integer DAY distances, not actual calendar dates.'
    p['store_product_pairs']=int(tx[['STORE_ID','PRODUCT_ID']].drop_duplicates().shape[0])
    demo=next((k for k in tables if 'demographic' in k),None)
    p['households_without_demographics']=p['customers']-tables[demo]['distinct_nonmissing']['household_key'] if demo else None
    # Full causal table audit streamed. Exact within-chunk duplicate count is clearly scoped.
    for member in z.namelist():
        if 'causal' not in member.lower() or not member.endswith('.csv'):continue
        n=0;missing=collections.Counter();dups=0;unique={};minmax={}
        with z.open(member) as f:
            for chunk in pd.read_csv(f,chunksize=500000):
                n+=len(chunk);missing.update({c:int(x) for c,x in chunk.isna().sum().items()})
                dups+=int(chunk.duplicated().sum())
                for c in chunk:
                    unique.setdefault(c,set()).update(chunk[c].dropna().unique().tolist())
                if n%5000000==0:print('Dunnhumby causal rows',n,flush=True)
        tables[Path(member).name]={'rows':n,'columns':len(unique),'missing':dict(missing),
            'distinct_nonmissing':{c:len(v) for c,v in unique.items()},
            'within_chunk_exact_duplicate_excess_rows_lower_bound':dups,
            'full_cross_chunk_duplicate_check':'not yet measured'}
    save('dunnhumby_profile.json',p);print('Dunnhumby complete',len(tx),flush=True)

def favorita():
    raw=ROOT/'data/raw/favorita'; tables={}
    for name in ['holidays_events','items','oil','stores','transactions']:
        path=raw/(name+'.csv')
        if not path.exists():subprocess.run(['bsdtar','-xf',str(path)+'.7z','-C',str(raw)],check=True)
        df=pd.read_csv(path);tables[name]=basic(df);tables[name]['file']=metadata(path)
    items=pd.read_csv(raw/'items.csv');stores=pd.read_csv(raw/'stores.csv')
    oil=pd.read_csv(raw/'oil.csv');hol=pd.read_csv(raw/'holidays_events.csv')
    test=pd.read_csv(raw/'test.csv');tables['test']=basic(test)
    tables['test'].update({'date_min':test.date.min(),'date_max':test.date.max(),
        'store_item_date_duplicate_excess':int(test.duplicated(['date','store_nbr','item_nbr']).sum()),
        'target_available':False})
    item_ids=sorted(items.item_nbr.tolist()); store_ids=sorted(stores.store_nbr.tolist())
    itmap={x:i for i,x in enumerate(item_ids)};stmap={x:i for i,x in enumerate(store_ids)}
    start=np.datetime64('2013-01-01','D');end=np.datetime64('2017-08-15','D');nd=int((end-start).astype(int))+1
    # Exact occupancy bit-array (bool): detects duplicate business keys across chunk boundaries.
    seen=np.zeros((nd,len(store_ids),len(item_ids)),dtype=bool)
    rows=0;missing=collections.Counter();counts=collections.Counter();daily=collections.Counter()
    distinct={c:set() for c in ['store_nbr','item_nbr','date','onpromotion']}
    prev_id=None;strict_id=True;keydups=0;unknown_items=0;unknown_stores=0
    minq=float('inf');maxq=float('-inf');mind=None;maxd=None
    first=np.full((len(store_ids),len(item_ids)),nd,dtype=np.int32)
    last=np.full_like(first,-1); entity_counts=np.zeros_like(first,dtype=np.int32)
    proc=subprocess.Popen(['bsdtar','-xOf',str(raw/'train.csv.7z'),'train.csv'],stdout=subprocess.PIPE)
    print('Favorita streaming full train.csv',flush=True)
    try:
        for chunk in pd.read_csv(proc.stdout,chunksize=500000,dtype={'id':'int64','store_nbr':'int16','item_nbr':'int32','unit_sales':'float64','onpromotion':'string'}):
            rows+=len(chunk);missing.update({c:int(x) for c,x in chunk.isna().sum().items()})
            ids=chunk.id.to_numpy();strict_id &= bool(np.all(np.diff(ids)==1) and (prev_id is None or ids[0]==prev_id+1));prev_id=int(ids[-1])
            for c in distinct:distinct[c].update(chunk[c].dropna().unique().tolist())
            counts['negative_unit_sales_rows']+=int(chunk.unit_sales.lt(0).sum())
            counts['zero_unit_sales_rows']+=int(chunk.unit_sales.eq(0).sum())
            counts['fractional_unit_sales_rows']+=int((chunk.unit_sales%1!=0).sum())
            counts['promotion_true_rows']+=int(chunk.onpromotion.str.lower().eq('true').sum())
            counts['promotion_false_rows']+=int(chunk.onpromotion.str.lower().eq('false').sum())
            minq=min(minq,float(chunk.unit_sales.min()));maxq=max(maxq,float(chunk.unit_sales.max()))
            cdmin=chunk.date.min();cdmax=chunk.date.max();mind=min(mind,cdmin) if mind else cdmin;maxd=max(maxd,cdmax) if maxd else cdmax
            daily.update({str(k):int(v) for k,v in chunk.date.value_counts().items()})
            si=chunk.store_nbr.map(stmap);ii=chunk.item_nbr.map(itmap)
            unknown_items+=int(ii.isna().sum());unknown_stores+=int(si.isna().sum())
            if si.isna().any() or ii.isna().any():raise ValueError('Unmapped keys; do not silently omit')
            di=(chunk.date.to_numpy(dtype='datetime64[D]')-start).astype(np.int32)
            if di.min()<0 or di.max()>=nd:raise ValueError('Date outside verified source scope')
            ss=si.to_numpy(dtype=np.int32);jj=ii.to_numpy(dtype=np.int32)
            flat=(di*len(store_ids)+ss)*len(item_ids)+jj
            uniq=np.unique(flat)
            keydups+=len(flat)-len(uniq)+int(seen.ravel()[uniq].sum());seen.ravel()[uniq]=True
            np.minimum.at(first,(ss,jj),di);np.maximum.at(last,(ss,jj),di)
            np.add.at(entity_counts,(ss,jj),1)
            if rows%5000000==0:print('Favorita train rows',rows,flush=True)
        if proc.wait()!=0:raise RuntimeError('7z decompression failed')
    finally:
        if proc.poll() is None:proc.terminate()
    occupied=seen.sum(axis=0,dtype=np.int32);active=last>=0
    p={'dataset':'Corporacion Favorita Grocery Sales Forecasting','status':'measured_full_train_stream',
       'rows':rows,'columns':6,'schema':['id','date','store_nbr','item_nbr','unit_sales','onpromotion'],
       'date_min':mind,'date_max':maxd,'calendar_days':nd,'days_with_records':len(distinct['date']),
       'missing_calendar_days':[str(d) for d in np.arange(start,end+np.timedelta64(1,'D')) if str(d) not in distinct['date']],
       'stores':len(distinct['store_nbr']),'products':len(distinct['item_nbr']),
       'item_master_rows':len(items),'families':int(items.family.nunique()),'classes':int(items['class'].nunique()),
       'customers':None,'missing':dict(missing),'counts':dict(counts),'unit_sales_min':minq,'unit_sales_max':maxq,
       'store_item_date_duplicate_excess_rows':keydups,'id_is_consecutive_in_file':strict_id,
       'exact_duplicate_excess_rows':0 if strict_id else None,
       'exact_duplicate_method':'Consecutive unique id proves zero identical full rows, including id.' if strict_id else 'Unverified',
       'item_orphan_rows':unknown_items,'store_orphan_rows':unknown_stores,'store_item_pairs':int(active.sum()),
       'observed_store_item_days':int(occupied.sum()),'full_master_grid_days':int(seen.size),
       'absent_master_grid_days':int(seen.size-occupied.sum()),
       'absent_days_within_first_last_observation':int(((last-first+1)[active]-occupied[active]).sum()),
       'zero_demand_caveat':'Explicit zeros counted. Missing records include inactive assortment and unobserved days; not automatically zeros.',
       'inventory':'not provided','price':'not provided','returns':'negative unit_sales proxy, not separately itemized returns',
       'oil_missing_rows':int(oil.dcoilwtico.isna().sum()),
       'holiday_transferred_rows':int(hol.transferred.sum()),
       'holiday_duplicate_dates':int(hol.duplicated(['date']).sum()),
       'tables':tables,'archive':metadata(raw/'favorita-grocery-sales-forecasting.zip'),
       'train_compressed':metadata(raw/'train.csv.7z')}
    # Only audit availability counts (not demand values) by date.
    save('favorita_daily_record_counts.json',dict(sorted(daily.items())))
    save('favorita_profile.json',p);print('Favorita complete',rows,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['uci','dunnhumby','favorita','combine']);args=parser.parse_args()
    if args.dataset=='combine':
        profiles={k:json.loads((OUT/f'{k}_profile.json').read_text()) for k in ['dunnhumby','favorita','uci']}
        save('dataset_profile.json',{'audit_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'scope':'Raw data quality audit only. No training, predictions, hybrid weights, or results.',
            'test_policy':'Full-file structural/data-quality counts inspected as requested; no test-based model or feature tuning.',
            'datasets':profiles})
        save('hardware.json',{'platform':platform.platform(),'machine':platform.machine(),'python':sys.version,
            'logical_cpu_count':os.cpu_count(),'ram_bytes':psutil.virtual_memory().total,
            'disk_free_bytes':psutil.disk_usage(str(ROOT)).free,'packages':{'pandas':pd.__version__,'numpy':np.__version__,'psutil':psutil.__version__}})
    else:globals()[args.dataset]()
