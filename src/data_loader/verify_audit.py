"""Additional exact cross-chunk and semantic checks; raw files remain unchanged."""
import json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'data/profiles'
z=zipfile.ZipFile(ROOT/'data/raw/dunnhumby/dunnhumby-complete-journey-data.zip')
p=json.loads((P/'dunnhumby_profile.json').read_text())
codes=np.empty(p['tables']['causal_data.csv']['rows'],dtype=np.uint64)
maps={'display':{},'mailer':{},'STORE_ID':{}};pos=0
with z.open('causal_data.csv') as f:
    for c in pd.read_csv(f,chunksize=500000,dtype={'display':'string','mailer':'string'}):
        for name in maps:
            for v in c[name].unique():
                if v not in maps[name]:maps[name][v]=len(maps[name])
            assert len(maps[name]) <= (1000 if name=='STORE_ID' else 16)
        assert c.WEEK_NO.between(0,999).all()
        assert c.PRODUCT_ID.between(0,100000000).all()
        base=(c.PRODUCT_ID.to_numpy(dtype=np.uint64)*1000+c.STORE_ID.map(maps['STORE_ID']).to_numpy(dtype=np.uint64))*1000+c.WEEK_NO.to_numpy(dtype=np.uint64)
        codes[pos:pos+len(c)]=base*256+c.display.map(maps['display']).to_numpy(dtype=np.uint64)*16+c.mailer.map(maps['mailer']).to_numpy(dtype=np.uint64)
        pos+=len(c)
assert pos==len(codes)
codes.sort()
t=p['tables']['causal_data.csv']
t['exact_duplicate_excess_rows']=int(np.sum(codes[1:]==codes[:-1]))
codes//=256
t['product_store_week_duplicate_excess_rows']=int(np.sum(codes[1:]==codes[:-1]))
t['full_cross_chunk_duplicate_check']='Complete; collision-free bounded integer encoding and full sort.'
del codes
for name in p['tables']:
    if name=='causal_data.csv':continue
    with z.open(name) as f:df=pd.read_csv(f,low_memory=False)
    p['tables'][name]['whitespace_only_values']={c:int(df[c].astype('string').str.fullmatch(r'\s*').fillna(False).sum()) for c in df.select_dtypes(include=['object','string'])}
    if name=='product.csv':
        p['tables'][name]['product_id_duplicate_excess']=int(df.PRODUCT_ID.duplicated().sum())
        p['tables'][name]['department_labels']=sorted(df.DEPARTMENT.unique().tolist())
(P/'dunnhumby_profile.json').write_text(json.dumps(p,indent=2)+'\n')
print('Dunnhumby exact checks complete',t,flush=True)

# No recalculation of demand, only join-key/availability checks on Favorita support tables.
raw=ROOT/'data/raw/favorita';aux={}
keys={'items':['item_nbr'],'stores':['store_nbr'],'oil':['date'],'transactions':['date','store_nbr']}
for name,key in keys.items():
    df=pd.read_csv(raw/(name+'.csv'))
    aux[name]={'join_key':key,'join_key_duplicate_excess_rows':int(df.duplicated(key).sum())}
    if 'date' in df:aux[name].update(date_min=df.date.min(),date_max=df.date.max())
    if name=='transactions':aux[name].update(zero_transaction_rows=int(df.transactions.eq(0).sum()),negative_transaction_rows=int(df.transactions.lt(0).sum()))
(P/'support_table_checks.json').write_text(json.dumps(aux,indent=2)+'\n')
