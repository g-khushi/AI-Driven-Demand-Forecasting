"""Four original test origins using frozen parameters; no optimizer or refitting."""
from pathlib import Path
import sys,json,gzip,pickle,hashlib,time,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor
from combined_v2.models import PooledLSTM,CrossModalTransformer
from combined_v2.hybrid import MODELS
from final_experiment.common import panels,inference,guard,R,M,E
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

def run():
    guard();assert not (R/'test_component_predictions.csv').exists(),'Test forecasts already saved; do not overwrite'
    record=json.loads((E/'training_complete.json').read_text())
    for p,h in record['model_checksums'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
    states=json.loads((M/'fold_1_transform_state.json').read_text());p=panels();output=[];costs=[];checks=[];failures=[]
    tree=XGBRegressor();tree.load_model(str(M/'fold_1_XGBoost.json'))
    nets={}
    for name in ['LSTM','Transformer']:
        saved=torch.load(M/f'fold_1_{name}.pt',map_location='cpu',weights_only=False)
        model=PooledLSTM(36) if name=='LSTM' else CrossModalTransformer(36);model.load_state_dict(saved['state_dict']);model.eval();nets[name]=model
    with gzip.open(M/'fold_1_SARIMA.pkl.gz','rb') as stream:sarima=pickle.load(stream)
    for fold in range(1,5):
        inputs,truth,entities=inference(p,states,fold);q,x,c,ids=inputs;scales=np.array([v[2] for v in entities])[:,None]
        for name in MODELS:
            t=time.perf_counter();pred=np.full((36,16),np.nan);fallback=0
            if name in ['Naive','SARIMA']:
                for i,(d,e,scale,origin,H) in enumerate(entities):
                    hist=p[d][p[d].entity.eq(e)&(p[d].t<=origin)].set_index('t').demand
                    if name=='Naive':
                        last=float(hist.dropna().iloc[-1])
                        for h in range(H):
                            value=hist.get(origin-6+h%7,np.nan)
                            if not np.isfinite(value):value=last;fallback+=1
                            pred[i,h]=value
                    elif d+'::'+e in sarima:
                        try:
                            original=sarima[d+'::'+e]
                            filtered=original if fold==1 else original.apply(hist.to_numpy(float),refit=False)
                            np.testing.assert_array_equal(original.params,filtered.params)
                            pred[i,:H]=filtered.forecast(H)
                        except Exception as exc:failures.append({'fold':fold,'dataset':d,'entity':e,'model':name,'error':repr(exc),'fallback':False})
            elif name=='XGBoost':pred=tree.predict(x.reshape(-1,x.shape[-1])).reshape(36,16)*scales
            else:
                with torch.no_grad():pred=nets[name](torch.tensor(q,dtype=torch.float32),torch.tensor(ids,dtype=torch.long),torch.tensor(c,dtype=torch.float32)).numpy()*scales
            elapsed=time.perf_counter()-t;clipped=int((pred<0).sum());pred=np.maximum(pred,0)
            for i,(d,e,scale,origin,H) in enumerate(entities):
                f=truth[truth.dataset.eq(d)&truth.entity.eq(e)].copy();f['prediction']=pred[i,:H];f['model']=name;output.append(f)
            costs.append({'fold':fold,'model':name,'inference_seconds':elapsed,'fallback_cells':fallback,'clipped_negative_forecasts':clipped,'parameter_updates':False})
        print('final-test origin',fold,'forecast with frozen parameters',flush=True)
    result=pd.concat(output,ignore_index=True)
    saved=pd.read_csv(R/'fold_1_component_predictions.csv',dtype={'entity':str})
    keys=['dataset','entity','model','horizon'];a=saved.sort_values(keys).prediction.to_numpy();b=result[result.fold.eq(1)].sort_values(keys).prediction.to_numpy()
    np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6,equal_nan=True)
    checks.append('Saved-model replay matches all first-origin forecasts within 1e-6')
    for pth,h in record['model_checksums'].items():assert hashlib.sha256((ROOT/pth).read_bytes()).hexdigest()==h
    checks.append('All model/transform artifacts unchanged across four test origins; SARIMA filter parameters exactly unchanged')
    result.to_csv(R/'test_component_predictions.csv',index=False);pd.DataFrame(costs).to_csv(R/'test_inference_costs.csv',index=False)
    (R/'forecast_verification.json').write_text(json.dumps({'status':'pass','checks':checks,'failures':failures},indent=2)+'\n')
    (E/'forecast_complete.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'predictions_sha256':hashlib.sha256((R/'test_component_predictions.csv').read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':run()
