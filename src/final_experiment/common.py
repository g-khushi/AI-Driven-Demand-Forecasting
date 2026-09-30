from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from combined_v2 import data
from feature_engineering.pilot_features import origin_features
from preprocessing.pilot_panels import tindex
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'experiments/final_experiment';R=ROOT/'results/final_experiment';M=ROOT/'models/final_experiment'
CFG=json.loads((E/'configuration_freeze.json').read_text());S=CFG['split'];DATASETS=['dunnhumby','favorita','uci']

def panels():
    return {d:pd.read_csv(ROOT/f'data/processed/final_experiment/{d}_panel.csv.gz',dtype={'entity':str,'store':str}).sort_values(['entity','t']).reset_index(drop=True) for d in DATASETS}

def assemble_training(p,unused_fold=1):
    previous=data.S
    replacement=json.loads(json.dumps(S))
    for d in DATASETS:replacement[d]['folds']['calibration']=replacement[d]['folds']['final_test']
    data.S=replacement
    try:arrays,inf,truth,states,proof=data.assemble(p,1)
    finally:data.S=previous
    truth['stage']='final_test';return arrays,inf,truth,states,proof

def inference(p,states,fold):
    inputs=[];truth=[];entities=[];offset=0
    for di,d in enumerate(DATASETS):
        origin=tindex(d,S[d]['folds']['final_test'][fold-1]['origin']);H=S[d]['horizon_days'];state=states[d]
        assert state['cutoff']==tindex(d,S[d]['folds']['final_test'][0]['origin'])
        for e in sorted(state['entities']):
            q,x,c,eid,provenance=origin_features(p[d],d,e,origin,H,state)
            assert provenance['max_input_timestamp']<=origin
            q=np.concatenate([q,np.broadcast_to(np.eye(3)[di],(28,3))],1)
            x=data.padded(x,0);x[:,40]+=offset;x=np.concatenate([x,np.broadcast_to(np.eye(3)[di],(16,3))],1);c=data.padded(c,0)
            inputs.append((q,x,c,eid+offset));entities.append((d,e,state['scales'][e],origin,H))
            hist=p[d][p[d].entity.eq(e)&(p[d].t<=origin)];regime='intermittent' if (hist.demand.dropna()>0).mean()<=.5 else 'frequent'
            for h in range(1,H+1):
                row=p[d][p[d].entity.eq(e)&p[d].t.eq(origin+h)].iloc[0]
                truth.append({'dataset':d,'entity':e,'series_id':d+'::'+e,'fold':fold,'stage':'final_test','origin':origin,'target_t':origin+h,'horizon':h,'y':row.demand,'score_mask':bool(row.score_mask),'weight':row.weight,'scale':state['scales'][e],'regime':regime,'volume':row.volume,'date':row.date})
        offset+=len(state['entities'])
    return [np.array([v[j] for v in inputs]) for j in range(4)],pd.DataFrame(truth),entities

def check_freeze():
    for path,h in CFG['source_checksums'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path

def guard():
    check_freeze();audit=json.loads((E/'leakage_audit.json').read_text());assert audit['status']=='pass'
    for path,h in audit['checksums'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
