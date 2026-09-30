"""Origin-bounded features, training-only transforms and masked sequence labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
BUSINESS=['promotion','oil','transactions','customers','holiday','price']
LOOKBACK=28

def load_panel(dataset):
    p=pd.read_csv(ROOT/f'data/processed/pilot/{dataset}_panel.csv.gz',dtype={'entity':str,'store':str})
    for c in ['observation_mask','score_mask','global_recording_mask','assumed_zero']:p[c]=p[c].astype(bool)
    return p.sort_values(['entity','t']).reset_index(drop=True)

def fit_state(panel,cutoff):
    train=panel[panel.t<=cutoff].copy()
    assert len(train) and train.t.max()<=cutoff
    scale=train.groupby('entity').demand.mean().clip(lower=1).fillna(1).to_dict()
    med={c:float(train[c].median()) if train[c].notna().any() else 0. for c in BUSINESS}
    std={c:max(float(train[c].std(ddof=0)),1.) if train[c].notna().any() else 1. for c in BUSINESS}
    return {'fit_max_timestamp':int(train.t.max()),'cutoff':int(cutoff),'scales':scale,
        'median':med,'std':std,'entities':{e:i for i,e in enumerate(sorted(train.entity.unique()))},
        'encoding':'entity ID only; no target encoding','imputation':'within-history forward fill then training-only median; missing target input=0 plus masks'}

def calendar(dataset,t):
    # Deterministic calendar known at origin; no realized future event table.
    if dataset=='dunnhumby':return [np.sin(2*np.pi*t/7),np.cos(2*np.pi*t/7),0.,0.]
    base=pd.Timestamp('2010-12-01' if dataset=='uci' else '2013-01-01');date=base+pd.Timedelta(days=int(t)-1)
    return [np.sin(2*np.pi*date.dayofweek/7),np.cos(2*np.pi*date.dayofweek/7),
            np.sin(2*np.pi*date.dayofyear/365.25),np.cos(2*np.pi*date.dayofyear/365.25)]

def origin_features(panel,dataset,entity,origin,horizon,state):
    h=panel[(panel.entity==entity)&(panel.t<=origin)].sort_values('t').copy()
    assert len(h) and int(h.t.max())<=origin
    scale=state['scales'][entity]
    # Reindex required context only; never read labels into features.
    seq=h.set_index('t').reindex(range(origin-LOOKBACK+1,origin+1))
    y=seq.demand.to_numpy(float)/scale
    channels=[np.nan_to_num(y,nan=0.),seq.observation_mask.fillna(False).to_numpy(float),seq.global_recording_mask.fillna(False).to_numpy(float)]
    for c in BUSINESS:
        series=h.set_index('t')[c].ffill().reindex(seq.index).ffill().fillna(state['median'][c])
        channels.append(((series.to_numpy(float)-state['median'][c])/state['std'][c]))
    sequence=np.stack(channels,axis=1).astype('float32')
    yd=h.set_index('t').demand
    lags=[float(yd.get(origin-k+1,np.nan))/scale for k in [1,2,3,7,14,28,56]]
    summaries=[]
    for window in [7,14,28,56]:
        x=h[h.t>origin-window].demand.dropna().to_numpy(float)/scale
        summaries.extend([np.mean(x),np.median(x),np.std(x),np.min(x),np.max(x)] if len(x) else [0.]*5)
    lag_missing=[float(not np.isfinite(v)) for v in lags]
    lags=np.nan_to_num(lags,nan=0.).tolist()
    latest_business=sequence[-1,3:].tolist()
    eid=state['entities'].get(entity,-1)
    # Unknown IDs cannot acquire labels/encodings from a future entity.
    assert eid>=0
    future=np.array([calendar(dataset,origin+j) for j in range(1,horizon+1)],dtype='float32')
    base=np.array(lags+lag_missing+summaries+latest_business+[float(eid),scale],dtype='float32')
    direct=np.array([np.concatenate([base,[j/horizon],future[j-1]]) for j in range(1,horizon+1)],dtype='float32')
    return sequence,direct,future,eid,{'max_input_timestamp':int(h.t.max()),'origin':int(origin),
        'scaler_fit_cutoff':state['cutoff'],'future_calendar':'deterministic known at origin'}

def training_arrays(panel,dataset,cutoff,horizon,state):
    seqs=[];direct=[];future=[];ids=[];targets=[];masks=[];provenance=[]
    # Chronological order, never random shuffle. Stride is fixed for resource pilot.
    first=int(panel.t.min())+55
    for origin in range(first,cutoff-horizon+1,7):
        for entity in sorted(state['entities']):
            hist=panel[(panel.entity==entity)&(panel.t<=origin)]
            if len(hist)<56:continue
            s,x,cal,eid,proof=origin_features(panel,dataset,entity,origin,horizon,state)
            label=panel[panel.entity.eq(entity)].set_index('t').reindex(range(origin+1,origin+horizon+1))
            assert origin+horizon<=cutoff
            mask=label.score_mask.fillna(False).to_numpy(bool)&label.demand.notna().to_numpy()
            if not mask.any():continue
            y=label.demand.fillna(0).to_numpy(float)/state['scales'][entity]
            seqs.append(s);direct.append(x);future.append(cal);ids.append(eid);targets.append(y);masks.append(mask)
            provenance.append({'entity':entity,'origin':origin,'max_input_timestamp':proof['max_input_timestamp'],
                'max_label_timestamp':origin+horizon,'training_cutoff':cutoff})
    assert seqs,'No valid training windows'
    return [np.array(x) for x in [seqs,direct,future,ids,targets,masks]],pd.DataFrame(provenance)
