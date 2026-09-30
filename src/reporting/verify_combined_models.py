"""Replay all saved pooled component forecasts, without refitting."""
from pathlib import Path
import sys,json,gzip,pickle,gc
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from combined_v2.models import PooledLSTM,CrossModalTransformer
from combined_v2.hybrid import MODELS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

def run():
    checks=[]
    for fold in range(1,5):
        p=pd.read_csv(ROOT/f'results/combined_v2/fold_{fold}_component_predictions.csv',dtype={'entity':str})
        rows=p[p.model.eq('Naive')];entities=list(rows[['dataset','entity','scale']].drop_duplicates().itertuples(index=False,name=None))
        inputs=np.load(ROOT/f'models/combined_v2/fold_{fold}_inference_inputs.npz');scales=np.array([v[2] for v in entities])[:,None]
        for name in MODELS:
            path=ROOT/f'models/combined_v2/fold_{fold}_{name}'
            if name=='Naive':
                a=json.loads(path.with_suffix('.json').read_text());pred=np.array([[a[d+'::'+e]['last_valid'] if a[d+'::'+e]['tail'][h%7] is None else a[d+'::'+e]['tail'][h%7] for h in range(16)] for d,e,_ in entities])
            elif name=='SARIMA':
                with gzip.open(str(path)+'.pkl.gz','rb') as stream:a=pickle.load(stream)
                pred=np.array([a[d+'::'+e].forecast(16) if d+'::'+e in a else np.full(16,np.nan) for d,e,_ in entities]);del a
            elif name=='XGBoost':
                a=XGBRegressor();a.load_model(str(path)+'.json');x=inputs['x'];pred=a.predict(x.reshape(-1,x.shape[-1])).reshape(36,16)*scales;del a
            else:
                saved=torch.load(str(path)+'.pt',map_location='cpu',weights_only=False);a=PooledLSTM(36) if name=='LSTM' else CrossModalTransformer(36);a.load_state_dict(saved['state_dict']);a.eval()
                with torch.no_grad():pred=a(torch.tensor(inputs['q'],dtype=torch.float32),torch.tensor(inputs['ids'],dtype=torch.long),torch.tensor(inputs['c'],dtype=torch.float32)).numpy()*scales
                del a,saved
            pred=np.maximum(pred,0);count=0
            for i,(d,e,_) in enumerate(entities):
                expected=p[p.model.eq(name)&p.dataset.eq(d)&p.entity.eq(e)].sort_values('horizon').prediction.to_numpy()
                np.testing.assert_allclose(pred[i,:len(expected)],expected,rtol=1e-6,atol=1e-6,equal_nan=True);count+=len(expected)
            checks.append({'fold':fold,'model':name,'cells':count,'status':'pass','tolerance':1e-6});gc.collect()
        print('fold',fold,'saved forecasts reproduced',flush=True)
    (ROOT/'results/combined_v2/saved_model_verification.json').write_text(json.dumps({'status':'pass','checks':checks},indent=2)+'\n')

if __name__=='__main__':run()
