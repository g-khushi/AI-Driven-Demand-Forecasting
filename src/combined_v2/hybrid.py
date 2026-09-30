"""Bounded convex Hybrid calibration on earlier out-of-sample forecasts only."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
MODELS=['Naive','SARIMA','XGBoost','LSTM','Transformer']
CANDIDATES=['inverse_MAE','pooled_MAE','dataset_shrinkage','regime_shrinkage']

def examples(frame):
    keys=['dataset','entity','fold','target_t']
    x=frame.pivot(index=keys,columns='model',values='prediction').reindex(columns=MODELS)
    y=frame.drop_duplicates(keys).set_index(keys).reindex(x.index)
    good=y.score_mask & y.y.notna() & x.notna().all(axis=1)
    return x.loc[good],y.loc[good]

def weights(x,y):
    # Equal dataset weight, normalized by each entity's training-only scale.
    n=len(y);k=len(MODELS)
    ds=y.index.get_level_values('dataset');counts=pd.Series(ds).value_counts()
    sample=np.array([1/counts[d] for d in ds]);sample/=sample.sum()
    a=x.to_numpy()/y.scale.to_numpy()[:,None];b=y.y.to_numpy()/y.scale.to_numpy()
    objective=np.r_[np.zeros(k),sample]
    from scipy.sparse import csr_matrix,eye,hstack,vstack
    constraints=vstack([hstack([csr_matrix(a),-eye(n)]),hstack([-csr_matrix(a),-eye(n)])],format='csr')
    result=linprog(objective,A_ub=constraints,b_ub=np.r_[b,-b],A_eq=np.r_[np.ones(k),np.zeros(n)][None,:],b_eq=[1.],bounds=[(.01,1.)]*k+[(0,None)]*n,method='highs')
    if not result.success:raise RuntimeError(result.message)
    w=result.x[:k];assert np.isclose(w.sum(),1) and (w>=.01-1e-8).all()
    return w

def fit(frame,method,max_fit_fold):
    assert method in CANDIDATES
    assert set(frame.fold.unique())<=set(range(1,max_fit_fold+1))
    assert (frame.target_t>frame.origin).all() and frame.stage.eq('calibration').all()
    x,y=examples(frame)
    pooled=weights(x,y);groups={};diagnostics={}
    for d in y.index.get_level_values('dataset').unique():
        mask=y.index.get_level_values('dataset')==d
        if method=='inverse_MAE':
            errors=np.abs(x.loc[mask].to_numpy()-y.loc[mask,'y'].to_numpy()[:,None]).mean(0)
            w=1/np.maximum(errors,1e-8);w/=w.sum()
        else:w=.75*weights(x.loc[mask],y.loc[mask])+.25*pooled
        groups[d]=w.tolist()
        if method=='regime_shrinkage':
            for regime in ['intermittent','frequent']:
                rm=mask & y.regime.eq(regime).to_numpy()
                # Predetermined support floor, not selected from later errors.
                if rm.sum()>=28:groups[d+'::'+regime]=(.5*weights(x.loc[rm],y.loc[rm])+.5*w).tolist()
        diagnostics[d]={'max_label':int(y.loc[mask].target_t.max()) if 'target_t' in y else int(y.loc[mask].index.get_level_values('target_t').max()),'cells':int(mask.sum())}
    return {'method':method,'max_fit_fold':max_fit_fold,'pooled_weights':pooled.tolist(),'groups':groups,'diagnostics':diagnostics,'component_order':MODELS}

def predict(frame,config):
    assert frame.fold.min()>config['max_fit_fold'],'Calibration/evaluation overlap'
    for d,g in frame.groupby('dataset'):
        assert g.origin.min()>=config['diagnostics'][d]['max_label'],'Future labels in weights'
    keys=['dataset','entity','fold','target_t'];x=frame.pivot(index=keys,columns='model',values='prediction').reindex(columns=MODELS)
    truth=frame.drop_duplicates(keys).set_index(keys).reindex(x.index)
    out=[]
    for key,row in truth.iterrows():
        d=key[0];method=config['method']
        w=config['pooled_weights'] if method=='pooled_MAE' else config['groups'].get(d+'::'+row.regime,config['groups'][d]) if method=='regime_shrinkage' else config['groups'][d]
        out.append(float(x.loc[key].to_numpy()@np.array(w)))
    truth['prediction']=out;truth['model']='Hybrid';return truth.reset_index()

def objective(frame):
    g=frame[frame.score_mask & frame.prediction.notna() & frame.y.notna()].copy()
    g['normalized_error']=(g.prediction-g.y).abs()/g.scale
    return float(g.groupby('dataset').normalized_error.mean().mean())
