"""Extend the unchanged pilot cohort through calibration and union retailer panels.

This is a 36-series bounded experiment, not a union of every raw transaction.
The old pilot and final-test boundary remain unchanged.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import pandas as pd
from preprocessing import pilot_panels as source
OUT=ROOT/'data/processed/combined_v2'
DATASETS=['dunnhumby','favorita','uci']

def prepare(dataset):
    source.OUT=OUT
    def bounds(d):
        s=source.SPLIT[d]
        return source.tindex(d,s['initial_training_end']),source.tindex(d,s['folds']['calibration'][-1]['forecast_end'])
    source.bounds=bounds
    getattr(source,dataset)()
    a=pd.read_csv(ROOT/f'data/processed/pilot/{dataset}_pilot_cohort.csv',dtype={'entity':str})
    b=pd.read_csv(OUT/f'{dataset}_pilot_cohort.csv',dtype={'entity':str})
    assert set(a.entity)==set(b.entity),'Cohort changed'

def combine():
    frames=[];manifest=[]
    for d in DATASETS:
        path=OUT/f'{d}_panel.csv.gz';p=pd.read_csv(path,dtype={'entity':str,'store':str})
        p['dataset']=d;p['series_id']=d+'::'+p.entity
        p['time_axis']='relative_DAY' if d=='dunnhumby' else 'calendar_date'
        cutoff=source.tindex(d,source.SPLIT[d]['initial_training_end'])
        train=p[p.t<=cutoff];scale=train.groupby('entity').demand.mean().clip(lower=1).fillna(1)
        p['initial_train_scale']=p.entity.map(scale);p['demand_initial_train_scaled']=p.demand/p.initial_train_scale
        p['partition']='initial_train'
        for stage in ['tuning','calibration']:
            for i,f in enumerate(source.SPLIT[d]['folds'][stage]):
                mask=p.t.between(source.tindex(d,f['forecast_start']),source.tindex(d,f['forecast_end']))
                p.loc[mask,'partition']=f'{stage}_{i+1}'
        test_start=source.tindex(d,source.SPLIT[d]['folds']['final_test'][0]['forecast_start'])
        assert p.t.max()<test_start and p.initial_train_scale.notna().all()
        frames.append(p)
        manifest.append({'dataset':d,'entities':p.entity.nunique(),'rows':len(p),'scorable_rows':int(p.score_mask.sum()),'initial_training_cutoff':cutoff,'development_end':int(p.t.max()),'final_test_start':test_start,'source_panel_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    out=pd.concat(frames,ignore_index=True,sort=False)
    assert out.series_id.nunique()==36 and not out.duplicated(['series_id','t']).any()
    out.to_csv(OUT/'combined_panel.csv.gz',index=False)
    result={'scope':'Existing 12 training-selected series per dataset; 36 total. Not full-population data.',
      'datasets':manifest,'combined_rows':len(out),'units':'Native units preserved. No cross-retailer summation. Training-only entity scaling.',
      'time':'Independent time axes. No invented Dunnhumby calendar dates. Pooling is retrospective multi-domain training, not synchronized real-time forecasting.',
      'final_test_included':False,'combined_sha256':hashlib.sha256((OUT/'combined_panel.csv.gz').read_bytes()).hexdigest()}
    (ROOT/'experiments/combined_v2/data_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('action',choices=DATASETS+['combine']);args=a.parse_args()
    combine() if args.action=='combine' else prepare(args.action)
