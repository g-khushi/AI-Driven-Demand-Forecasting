"""Small CPU pilot on fixed calibration origins, protected by checksum-bound tests."""
from pathlib import Path
import os,sys,json,time,hashlib,threading,pickle,gc,warnings,traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from feature_engineering.pilot_features import load_panel,fit_state,origin_features,training_arrays
from preprocessing.pilot_panels import tindex
from hybrid.hybrid_model import MODELS,fit_weights,combine
from evaluation.metrics import metrics
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
CFG=json.loads((ROOT/'experiments/continuation_protocol.json').read_text())
SPLITS=json.loads((ROOT/'experiments/split_manifest.json').read_text())['datasets']
PRED=ROOT/'predictions/pilot';RESULT=ROOT/'results/pilot';MODEL=ROOT/'models/pilot'
for d in [PRED,RESULT,MODEL]:d.mkdir(parents=True,exist_ok=True)

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')
def guard():
    audit=json.loads((ROOT/'experiments/leakage_audit_report.json').read_text())
    assert audit['status']=='pass','Leakage tests must pass before any model training'
    for p,sha in audit['checksums'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,'Stale leakage test: '+p
    assert not CFG['final_test_enabled'] and not CFG['full_training_enabled']

class Monitor:
    def __enter__(self):
        self.stop=False;self.base=psutil.Process().memory_info().rss;self.peak=self.base
        def watch():
            while not self.stop:
                p=psutil.Process();rss=p.memory_info().rss
                for c in p.children(recursive=True):
                    try:rss+=c.memory_info().rss
                    except psutil.Error:pass
                self.peak=max(self.peak,rss);time.sleep(.02)
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop=True;self.thread.join()

class SequenceModel(nn.Module):
    def __init__(self,kind,channels,H,nentities):
        super().__init__();self.kind=kind;self.embed=nn.Embedding(nentities,8)
        if kind=='LSTM':self.encoder=nn.LSTM(channels,32,num_layers=1,batch_first=True)
        else:
            self.input_projection=nn.Linear(channels,32)
            layer=nn.TransformerEncoderLayer(d_model=32,nhead=2,dim_feedforward=64,dropout=.1,batch_first=True)
            self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
            pos=torch.arange(28)[:,None];freq=torch.exp(torch.arange(0,32,2)*(-np.log(10000.)/32))
            pe=torch.zeros(28,32);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
            self.register_buffer('position',pe)
        self.head=nn.Linear(32+8+H*4,H)
    def forward(self,x,entities,future_calendar):
        if self.kind=='LSTM':out,_=self.encoder(x)
        else:out=self.encoder(self.input_projection(x)+self.position)
        return self.head(torch.cat([out[:,-1],self.embed(entities),future_calendar.flatten(1)],dim=1))

def run_dataset(dataset):
    guard();p=load_panel(dataset);H=SPLITS[dataset]['horizon_days'];allpred=[];measure=[];failures=[];trainlogs=[]
    states={}
    for fi,fold in enumerate(SPLITS[dataset]['folds']['calibration'][:2]):
        stage='calibration_fit' if fi==0 else 'pilot_validation';origin=tindex(dataset,fold['origin']);state=fit_state(p,origin);states[stage]=state
        arrays,proof=training_arrays(p,dataset,origin,H,state);seq,X,future,ids,Y,M=arrays
        entities=sorted(state['entities']);inference=[origin_features(p,dataset,e,origin,H,state) for e in entities]
        iq=np.array([v[0] for v in inference]);ix=np.array([v[1] for v in inference]);ical=np.array([v[2] for v in inference]);ie=np.array([v[3] for v in inference])
        xflat=X.reshape(-1,X.shape[-1]);yf=Y.ravel();mf=M.ravel()
        tens=[torch.tensor(a,dtype=torch.long if j==3 else torch.float32) for j,a in enumerate([seq,X,future,ids,Y,M])]
        itensors=[torch.tensor(iq,dtype=torch.float32),torch.tensor(ie,dtype=torch.long),torch.tensor(ical,dtype=torch.float32)]
        actual=p.set_index(['entity','t'])
        for model_name in MODELS:
            guard();print(dataset,stage,model_name,flush=True)
            torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed'])
            savepath=MODEL/f'{dataset}_{stage}_{model_name}';artifact={};params=None;param_kind=None;fallback=0;warning_list=[]
            failed_entities=set();pred=np.full((len(entities),H),np.nan);train_s=0.;infer_s=0.;latencies=[]
            with Monitor() as mon:
                try:
                    if model_name=='Naive':
                        t=time.perf_counter();last={e:p[(p.entity==e)&(p.t<=origin)].set_index('t').demand for e in entities};train_s=time.perf_counter()-t
                        t=time.perf_counter()
                        for i,e in enumerate(entities):
                            hist=last[e];lastvalid=float(hist.dropna().iloc[-1]);v=[]
                            for h in range(1,H+1):
                                value=hist.get(origin-6+(h-1)%7,np.nan)
                                if not np.isfinite(value):value=lastvalid;fallback+=1
                                v.append(value)
                            pred[i]=v
                        infer_s=time.perf_counter()-t;artifact={e:{'seasonal_tail':{int(k):(float(z) if np.isfinite(z) else None) for k,z in v.tail(7).items()},'last_valid':float(v.dropna().iloc[-1])} for e,v in last.items()}
                        params=0;param_kind='learned_coefficients'
                        dump(savepath.with_suffix('.json'),{'seasonal_period':7,'state':artifact,'fallback_policy':'last historical valid target for missing seasonal lag','fallback_cells':fallback})
                    elif model_name=='SARIMA':
                        fitted={};t=time.perf_counter()
                        for e in entities:
                            y=p[(p.entity==e)&(p.t<=origin)].demand.to_numpy(float)
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                fit=SARIMAX(y,order=(1,0,0),seasonal_order=(1,0,0,7),trend='c',enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=100)
                            warning_list.extend(str(w.message) for w in caught)
                            if not fit.mle_retvals.get('converged',False):
                                failed_entities.add(e);failures.append({'dataset':dataset,'model':model_name,'stage':stage,'entity':e,'reason':'optimizer_not_converged','fallback':False})
                            else:fitted[e]=fit
                        train_s=time.perf_counter()-t;t=time.perf_counter()
                        for i,e in enumerate(entities):
                            if e in fitted:pred[i]=fitted[e].forecast(H)
                        infer_s=time.perf_counter()-t;params=sum(len(v.params) for v in fitted.values());param_kind='fitted_coefficients'
                        with savepath.with_suffix('.pkl').open('wb') as out:pickle.dump(fitted,out)
                    elif model_name=='XGBoost':
                        model=XGBRegressor(n_estimators=100,max_depth=3,learning_rate=.05,objective='reg:squarederror',tree_method='hist',subsample=1.,colsample_bytree=1.,n_jobs=2,random_state=CFG['seed'])
                        t=time.perf_counter();model.fit(xflat[mf],yf[mf]);train_s=time.perf_counter()-t
                        t=time.perf_counter();pred=model.predict(ix.reshape(-1,ix.shape[-1])).reshape(len(entities),H);infer_s=time.perf_counter()-t
                        pred*=np.array([state['scales'][e] for e in entities])[:,None]
                        model.save_model(str(savepath.with_suffix('.json')));params=len(model.get_booster().trees_to_dataframe());param_kind='tree_nodes_not_neural_parameters'
                    else:
                        model=SequenceModel(model_name,seq.shape[-1],H,len(entities));optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                        t=time.perf_counter();model.train()
                        for epoch in range(CFG['neural_epochs']):
                            losses=[]
                            for a in range(0,len(seq),CFG['neural_batch_size']):
                                b=a+CFG['neural_batch_size'];optimizer.zero_grad()
                                out=model(tens[0][a:b],tens[3][a:b],tens[2][a:b]);mask=tens[5][a:b]
                                loss=(((out-tens[4][a:b])**2)*mask).sum()/mask.sum().clamp(min=1)
                                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                            trainlogs.append({'dataset':dataset,'stage':stage,'model':model_name,'epoch':epoch+1,'masked_scaled_MSE':float(np.mean(losses))})
                        train_s=time.perf_counter()-t;model.eval();t=time.perf_counter()
                        with torch.no_grad():pred=model(*itensors).numpy()
                        infer_s=time.perf_counter()-t;pred*=np.array([state['scales'][e] for e in entities])[:,None]
                        params=sum(v.numel() for v in model.parameters());param_kind='trainable_parameters'
                        torch.save({'state_dict':model.state_dict(),'kind':model_name,'features':seq.shape[-1],'horizon':H,'nentities':len(entities),'transform_state':state},savepath.with_suffix('.pt'))
                    negatives=int((pred<0).sum());pred=np.maximum(pred,0.)
                except Exception as exc:
                    negatives=0;failed_entities=set(entities);failures.append({'dataset':dataset,'stage':stage,'model':model_name,'reason':repr(exc),'traceback':traceback.format_exc(),'fallback':False});pred[:]=np.nan
            files=list(MODEL.glob(savepath.name+'.*'));size=sum(x.stat().st_size for x in files)
            measure.append({'dataset':dataset,'stage':stage,'model':model_name,'training_seconds':train_s,'inference_seconds':infer_s,
                'latency_seconds_per_entity':infer_s/len(entities),'peak_RSS_bytes':mon.peak,'peak_incremental_RSS_bytes':mon.peak-mon.base,
                'parameters':params,'parameter_kind':param_kind,'model_bytes':size,'gpu_memory_bytes':None,'device':'CPU','seed':CFG['seed'],
                'successful_entities':len(entities)-len(failed_entities),'total_entities':len(entities),'fallback_cells':fallback,'negative_forecasts_clipped':negatives,
                'memory_note':'20ms RSS sampling; sequential shared process; absolute RSS includes imported libraries and retained allocator pages'})
            if warning_list:dump(ROOT/f'logs/pilot/{dataset}_{stage}_{model_name}_warnings.json',warning_list)
            for i,e in enumerate(entities):
                for h in range(1,H+1):
                    row=actual.loc[(e,origin+h)]
                    allpred.append({'dataset':dataset,'stage':stage,'model':model_name,'origin':origin,'entity':e,'horizon':h,'target_t':origin+h,
                        'date':row.date,'y':row.demand,'prediction':float(pred[i,h-1]),'score_mask':bool(row.score_mask),
                        'weight':float(row.weight),'volume':row.volume,'category':row.get('category','unknown'),'store':row.get('store','unknown'),
                        'promotion':row.promotion,'holiday':row.holiday,'assumed_zero':bool(row.assumed_zero),'observation_mask':bool(row.observation_mask),
                        'returned_units':row.get('returned_units',0.),'status':'failed' if e in failed_entities else 'success'})
            gc.collect()
        dump(MODEL/f'{dataset}_{stage}_transform_state.json',state)
    frame=pd.DataFrame(allpred);first=frame[frame.stage.eq('calibration_fit')];second=frame[frame.stage.eq('pilot_validation')]
    origin=int(second.origin.iloc[0]);t=time.perf_counter();weights=fit_weights(first,origin);weight_s=time.perf_counter()-t
    dump(MODEL/f'{dataset}_Hybrid_weights.json',weights)
    t=time.perf_counter();hybrid=combine(second,weights);inference_s=time.perf_counter()-t
    base=second[second.model.eq('Naive')].set_index(['entity','target_t']).copy();base['prediction']=hybrid;base['model']='Hybrid';base['status']=np.where(base.prediction.notna(),'success','failed')
    frame=pd.concat([frame,base.reset_index()],ignore_index=True)
    cost=pd.DataFrame(measure);evaluation=cost[cost.stage.eq('pilot_validation')]
    cost=pd.concat([cost,pd.DataFrame([{'dataset':dataset,'stage':'pilot_validation','model':'Hybrid',
        'training_seconds':float(cost.training_seconds.sum()+weight_s),'inference_seconds':float(evaluation.inference_seconds.sum()+inference_s),
        'marginal_weight_fit_seconds':weight_s,'marginal_combination_seconds':inference_s,
        'peak_RSS_bytes':int(cost.peak_RSS_bytes.max()),'model_bytes':int(evaluation.model_bytes.sum())+(MODEL/f'{dataset}_Hybrid_weights.json').stat().st_size,
        'parameter_kind':'heterogeneous component parameters plus five scalar weights','parameters':None,
        'device':'CPU','seed':CFG['seed'],'successful_entities':int(base.groupby(level=0).prediction.apply(lambda x:x.notna().all()).sum()),
        'total_entities':len(entities),'memory_note':'Maximum component RSS; end-to-end ensemble memory not benchmarked concurrently; training includes calibration component fits and validation refits.'}])],ignore_index=True)
    frame.to_csv(PRED/f'{dataset}_predictions.csv',index=False);cost.to_csv(RESULT/f'{dataset}_compute.csv',index=False)
    dump(RESULT/f'{dataset}_failures.json',failures);pd.DataFrame(trainlogs).to_csv(ROOT/f'logs/pilot/{dataset}_training.csv',index=False)
    dump(MODEL/f'{dataset}_configuration.json',{'pilot_config':CFG,'split':SPLITS[dataset],'weights':weights,'final_test_used':False,
        'neural_architecture':'one-layer hidden32 LSTM or one-layer dim32 heads2 Transformer with entity embedding8, calendar input and direct H-output head',
        'sarima':{'order':[1,0,0],'seasonal_order':[1,0,0,7],'trend':'constant','maxiter':100},'forecast_nonnegative_clip':True})
    print(dataset,'finished pilot',flush=True)

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('dataset',choices=list(SPLITS));arg=a.parse_args();run_dataset(arg.dataset)
