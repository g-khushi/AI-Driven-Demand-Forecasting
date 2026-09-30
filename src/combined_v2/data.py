from pathlib import Path
import json
import numpy as np
import pandas as pd
from feature_engineering.pilot_features import fit_state,origin_features,training_arrays
from preprocessing.pilot_panels import tindex
ROOT=Path(__file__).resolve().parents[2]
DATASETS=['dunnhumby','favorita','uci'];HMAX=16
S=json.loads((ROOT/'experiments/split_manifest.json').read_text())['datasets']

def panels():
    result={}
    for d in DATASETS:
        p=pd.read_csv(ROOT/f'data/processed/combined_v2/{d}_panel.csv.gz',dtype={'entity':str,'store':str})
        result[d]=p.sort_values(['entity','t']).reset_index(drop=True)
    return result

def padded(a,axis=1):
    pads=[(0,0)]*a.ndim;pads[axis]=(0,HMAX-a.shape[axis]);return np.pad(a,pads)

def assemble(allpanels,fold):
    batches=[];inference=[];truths=[];states={};proofs=[];offset=0
    for di,d in enumerate(DATASETS):
        p=allpanels[d];H=S[d]['horizon_days'];origin=tindex(d,S[d]['folds']['calibration'][fold-1]['origin'])
        state=fit_state(p,origin);states[d]=state
        arrays,proof=training_arrays(p,d,origin,H,state)
        q,x,c,ids,y,m=arrays
        domain=np.zeros((len(q),28,3),dtype=np.float32);domain[:,:,di]=1
        q=np.concatenate([q,domain],2);x=padded(x);c=padded(c);y=padded(y);m=padded(m)
        # Direct tree features include dataset ID as well as globally unique entity ID.
        x[:,:,40]+=offset
        x=np.concatenate([x,np.broadcast_to(np.eye(3)[di],(*x.shape[:2],3))],2)
        proof['dataset']=d;proof['fold']=fold;proofs.append(proof)
        # Domain-balanced masked squared error; weights sum equally per source.
        sample_weight=np.full(len(q),1/max(1,int(m.sum())),dtype=np.float32)
        batches.append((q,x,c,ids+offset,y,m,sample_weight))
        for e in sorted(state['entities']):
            a,b,cal,eid,_=origin_features(p,d,e,origin,H,state)
            dom=np.broadcast_to(np.eye(3)[di],(28,3));a=np.concatenate([a,dom],1)
            b=padded(b,0);b[:,40]+=offset;b=np.concatenate([b,np.broadcast_to(np.eye(3)[di],(16,3))],1);cal=padded(cal,0)
            inference.append((a,b,cal,eid+offset))
            hist=p[(p.entity==e)&(p.t<=origin)];regime='intermittent' if (hist.demand.dropna()>0).mean()<=.5 else 'frequent'
            for h in range(1,H+1):
                row=p[p.entity.eq(e)&p.t.eq(origin+h)].iloc[0]
                truths.append({'dataset':d,'entity':e,'series_id':d+'::'+e,'fold':fold,'stage':'calibration','origin':origin,'target_t':origin+h,'horizon':h,'y':row.demand,'score_mask':bool(row.score_mask),'weight':row.weight,'scale':state['scales'][e],'regime':regime,'volume':row.volume,'date':row.date})
        offset+=len(state['entities'])
    arrays=[np.concatenate([b[j] for b in batches],axis=0) for j in range(7)]
    # Interleave domains by relative progress, preserving each domain's chronology.
    proof=pd.concat(proofs,ignore_index=True);progress=proof.origin.to_numpy()/proof.training_cutoff.to_numpy()
    order=np.argsort(progress,kind='stable');arrays=[a[order] for a in arrays];proof=proof.iloc[order].reset_index(drop=True)
    inf=[np.array([v[j] for v in inference]) for j in range(4)]
    assert (proof.max_input_timestamp<=proof.origin).all() and (proof.max_label_timestamp<=proof.training_cutoff).all()
    return arrays,inf,pd.DataFrame(truths),states,proof
