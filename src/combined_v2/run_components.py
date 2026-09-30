"""Four pooled development folds, five permitted Hybrid components, no final test."""
from pathlib import Path
import sys,os,json,time,gzip,pickle,gc,hashlib,threading,warnings
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import torch
import psutil
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from combined_v2.data import panels,assemble,DATASETS,S
from combined_v2.models import PooledLSTM,CrossModalTransformer
from combined_v2.hybrid import MODELS
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
OUT=ROOT/'results/combined_v2';ART=ROOT/'models/combined_v2'

def guard():
    a=json.loads((ROOT/'experiments/combined_v2/leakage_audit.json').read_text());assert a['status']=='pass'
    for path,h in a['checksums'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path

class RSS:
    def __enter__(self):
        self.peak=psutil.Process().memory_info().rss;self.stop=False
        def f():
            while not self.stop:self.peak=max(self.peak,psutil.Process().memory_info().rss);time.sleep(.02)
        self.thread=threading.Thread(target=f,daemon=True);self.thread.start();return self
    def __exit__(self,*a):self.stop=True;self.thread.join()

def run(fold):
    guard();cfg=json.loads((ROOT/'experiments/combined_v2/protocol.json').read_text());allpanels=panels()
    arrays,inf,truth,states,proof=assemble(allpanels,fold)
    q,x,c,ids,y,m,sample_weight=arrays;iq,ix,ic,ii=inf
    proof.to_csv(OUT/f'fold_{fold}_training_provenance.csv.gz',index=False)
    (ART/f'fold_{fold}_transform_state.json').write_text(json.dumps(states,indent=2))
    np.savez_compressed(ART/f'fold_{fold}_inference_inputs.npz',q=iq,x=ix,c=ic,ids=ii)
    entities=list(truth[['dataset','entity','scale']].drop_duplicates().itertuples(index=False,name=None));assert len(entities)==36
    rows=[];costs=[];failures=[];losses=[]
    for name in MODELS:
        guard();torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);pred=np.full((36,16),np.nan);fit_s=0.;infer_s=0.;params=None;fallback=0
        path=ART/f'fold_{fold}_{name}';print('fold',fold,name,flush=True)
        with RSS() as mon:
            started=time.perf_counter()
            if name=='Naive':
                fitted={}
                for i,(d,e,scale) in enumerate(entities):
                    origin=states[d]['cutoff'];g=allpanels[d];history=g[g.entity.eq(e)&(g.t<=origin)].set_index('t').demand;last=float(history.dropna().iloc[-1]);tail=history.reindex(range(origin-6,origin+1))
                    fitted[d+'::'+e]={'tail':[float(v) if np.isfinite(v) else None for v in tail],'last_valid':last}
                fit_s=time.perf_counter()-started;started=time.perf_counter()
                for i,(d,e,scale) in enumerate(entities):
                    state=fitted[d+'::'+e]
                    for h in range(S[d]['horizon_days']):
                        v=state['tail'][h%7]
                        if v is None:v=state['last_valid'];fallback+=1
                        pred[i,h]=v
                infer_s=time.perf_counter()-started;params=0;path.with_suffix('.json').write_text(json.dumps(fitted))
            elif name=='SARIMA':
                fitted={}
                for i,(d,e,scale) in enumerate(entities):
                    origin=states[d]['cutoff'];g=allpanels[d];history=g[g.entity.eq(e)&(g.t<=origin)].demand.to_numpy(float)
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');model=SARIMAX(history,order=(1,0,0),seasonal_order=(1,0,0,7),trend='c').fit(disp=False,maxiter=100)
                        if not model.mle_retvals.get('converged',False):raise RuntimeError('SARIMA nonconvergence')
                        fitted[d+'::'+e]=model
                        for w in caught:failures.append({'fold':fold,'model':name,'dataset':d,'entity':e,'type':'warning','message':str(w.message)})
                    except Exception as exc:failures.append({'fold':fold,'model':name,'dataset':d,'entity':e,'type':'failed','message':repr(exc),'fallback':False})
                fit_s=time.perf_counter()-started;started=time.perf_counter()
                for i,(d,e,scale) in enumerate(entities):
                    if d+'::'+e in fitted:pred[i,:S[d]['horizon_days']]=fitted[d+'::'+e].forecast(S[d]['horizon_days'])
                infer_s=time.perf_counter()-started;params=sum(len(v.params) for v in fitted.values())
                with gzip.open(str(path)+'.pkl.gz','wb',compresslevel=3) as stream:pickle.dump(fitted,stream)
                del fitted
            elif name=='XGBoost':
                model=XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.05,objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=cfg['seed'])
                flat=x.reshape(-1,x.shape[-1]);valid=m.ravel();sw=np.repeat(sample_weight,16);sw=sw/sw[valid].mean()
                model.fit(flat[valid],y.ravel()[valid],sample_weight=sw[valid]);fit_s=time.perf_counter()-started;started=time.perf_counter()
                pred=model.predict(ix.reshape(-1,ix.shape[-1])).reshape(36,16)*np.array([v[2] for v in entities])[:,None];infer_s=time.perf_counter()-started
                model.save_model(str(path)+'.json');params=len(model.get_booster().trees_to_dataframe());del model
            else:
                model=PooledLSTM(36) if name=='LSTM' else CrossModalTransformer(36)
                optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
                tensors=[torch.tensor(v,dtype=torch.long if j==2 else torch.float32) for j,v in enumerate([q,c,ids,y,m,sample_weight])]
                started=time.perf_counter();model.train()
                for epoch in range(cfg['neural_epochs']):
                    numerator=0.;denominator=0.
                    for start in range(0,len(q),64):
                        end=start+64;optimizer.zero_grad();output=model(tensors[0][start:end],tensors[2][start:end],tensors[1][start:end]);weights=tensors[4][start:end]*tensors[5][start:end,None]
                        # Constant full-epoch normalization preserves domain weighting between batches.
                        value=((output-tensors[3][start:end])**2*weights).sum()
                        loss=value*(len(q)/64)/3;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        numerator+=float(value.detach());denominator+=float(weights.sum())
                    losses.append({'fold':fold,'model':name,'epoch':epoch+1,'domain_balanced_scaled_MSE':numerator/denominator})
                fit_s=time.perf_counter()-started;model.eval();started=time.perf_counter()
                with torch.no_grad():pred=model(torch.tensor(iq,dtype=torch.float32),torch.tensor(ii,dtype=torch.long),torch.tensor(ic,dtype=torch.float32)).numpy()*np.array([v[2] for v in entities])[:,None]
                infer_s=time.perf_counter()-started;params=sum(v.numel() for v in model.parameters());torch.save({'state_dict':model.state_dict(),'model':name,'nentities':36,'configuration':cfg},str(path)+'.pt');del model,optimizer,tensors
            clipped=int(np.sum(pred<0));pred=np.maximum(pred,0)
        for i,(d,e,scale) in enumerate(entities):
            f=truth[truth.dataset.eq(d)&truth.entity.eq(e)].copy();f['prediction']=pred[i,:len(f)];f['model']=name;rows.append(f)
        size=sum(p.stat().st_size for p in ART.glob(path.name+'.*'))
        costs.append({'fold':fold,'model':name,'training_seconds':fit_s,'inference_seconds':infer_s,'peak_process_RSS_bytes':mon.peak,'artifact_bytes':size,'parameters_or_tree_nodes':params,'fallback_cells':fallback,'negative_forecasts_clipped':clipped,'scope':'36-series pooled development; CPU2; own-process RSS; feature building and serialization excluded from fit timing'})
        gc.collect()
    pd.concat(rows).to_csv(OUT/f'fold_{fold}_component_predictions.csv',index=False);pd.DataFrame(costs).to_csv(OUT/f'fold_{fold}_compute.csv',index=False);pd.DataFrame(losses).to_csv(OUT/f'fold_{fold}_training.csv',index=False)
    (OUT/f'fold_{fold}_failures.json').write_text(json.dumps(failures,indent=2));print('fold',fold,'complete',flush=True)

if __name__=='__main__':run(int(sys.argv[1]))
