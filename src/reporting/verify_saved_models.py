"""Reload trusted local artifacts and independently reproduce second-fold forecasts."""
from pathlib import Path
import sys,json,pickle,gc
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from evaluation.run_pilot import SequenceModel
from feature_engineering.pilot_features import load_panel,origin_features
from hybrid.hybrid_model import MODELS

def run():
    checks=[]
    for dataset in ['dunnhumby','favorita','uci']:
        p=load_panel(dataset)
        f=pd.read_csv(ROOT/f'predictions/pilot/{dataset}_predictions.csv',dtype={'entity':str})
        f=f[f.stage.eq('pilot_validation')];origin=int(f.origin.iloc[0]);H=int(f.horizon.max())
        state=json.loads((ROOT/f'models/pilot/{dataset}_pilot_validation_transform_state.json').read_text())
        entities=sorted(state['entities']);inputs=[origin_features(p,dataset,e,origin,H,state) for e in entities]
        iq=np.array([v[0] for v in inputs]);ix=np.array([v[1] for v in inputs]);ical=np.array([v[2] for v in inputs]);ie=np.array([v[3] for v in inputs])
        scales=np.array([state['scales'][e] for e in entities])[:,None];preds={}
        for name in MODELS:
            base=ROOT/f'models/pilot/{dataset}_pilot_validation_{name}'
            expected=f[f.model.eq(name)].pivot(index='entity',columns='horizon',values='prediction').reindex(entities).to_numpy()
            if name=='Naive':
                artifact=json.loads(base.with_suffix('.json').read_text());pred=[]
                for e in entities:
                    row=[];s=artifact['state'][e]
                    for h in range(1,H+1):
                        v=s['seasonal_tail'].get(str(origin-6+(h-1)%7))
                        row.append(s['last_valid'] if v is None else v)
                    pred.append(row)
                pred=np.asarray(pred)
            elif name=='SARIMA':
                with base.with_suffix('.pkl').open('rb') as stream:model=pickle.load(stream)
                pred=np.array([model[e].forecast(H) if e in model else np.full(H,np.nan) for e in entities])
                del model
            elif name=='XGBoost':
                model=XGBRegressor();model.load_model(str(base.with_suffix('.json')))
                pred=model.predict(ix.reshape(-1,ix.shape[-1])).reshape(len(entities),H)*scales;del model
            else:
                artifact=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False)
                model=SequenceModel(name,artifact['features'],H,len(entities));model.load_state_dict(artifact['state_dict']);model.eval()
                with torch.no_grad():pred=model(torch.tensor(iq,dtype=torch.float32),torch.tensor(ie,dtype=torch.long),torch.tensor(ical,dtype=torch.float32)).numpy()*scales
                del model,artifact
            pred=np.maximum(pred,0)
            np.testing.assert_allclose(pred,expected,rtol=1e-6,atol=1e-6,equal_nan=True)
            preds[name]=pred;checks.append({'dataset':dataset,'model':name,'status':'pass','cells':int(np.isfinite(expected).sum()),'tolerance':1e-6})
            gc.collect()
        w=json.loads((ROOT/f'models/pilot/{dataset}_Hybrid_weights.json').read_text())['weights']
        hybrid=sum(preds[m]*w[m] for m in MODELS)
        expected=f[f.model.eq('Hybrid')].pivot(index='entity',columns='horizon',values='prediction').reindex(entities).to_numpy()
        np.testing.assert_allclose(hybrid,expected,rtol=1e-6,atol=1e-6,equal_nan=True)
        checks.append({'dataset':dataset,'model':'Hybrid','status':'pass','cells':int(np.isfinite(expected).sum()),'tolerance':1e-6})
        print(dataset,'saved-model forecast replay passed',flush=True)
    result={'status':'pass','checks':checks,'scope':'All second-fold forecasts including masked dates; no fitting; trusted local artifacts only'}
    (ROOT/'results/pilot/saved_model_replay.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':run()
