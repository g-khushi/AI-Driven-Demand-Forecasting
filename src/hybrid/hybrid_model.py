"""New transparent five-component Hybrid; no legacy code was present."""
import numpy as np
MODELS=['Naive','SARIMA','XGBoost','LSTM','Transformer']

def assert_calibration_boundary(frame,evaluation_origin):
    if len(frame)==0:raise ValueError('Empty calibration evidence')
    if not frame.stage.eq('calibration_fit').all():raise ValueError('Weights require calibration_fit predictions only')
    if int(frame.target_t.max())>evaluation_origin:raise ValueError('Calibration labels after evaluation origin')
    if (frame.origin>=evaluation_origin).any():raise ValueError('Cannot fit weights on evaluation-origin forecasts')
    if not (frame.target_t>frame.origin).all():raise ValueError('Not out-of-sample predictions')

def fit_weights(frame,evaluation_origin,models=None):
    models=MODELS if models is None else models
    assert set(models)<=set(MODELS) and len(models)>=2
    assert_calibration_boundary(frame,evaluation_origin)
    pivot=frame.pivot(index=['entity','target_t'],columns='model',values='prediction')
    truth=frame.drop_duplicates(['entity','target_t']).set_index(['entity','target_t'])
    valid=truth.score_mask.reindex(pivot.index).fillna(False)&pivot[models].notna().all(axis=1)
    if not valid.any():raise ValueError('No common successful calibration cells')
    y=truth.y.reindex(pivot.index)[valid].to_numpy(float)
    errors=np.array([np.mean(np.abs(pivot.loc[valid,m].to_numpy()-y)) for m in models])
    eps=1e-8*max(1.,float(np.mean(np.abs(y))))
    if (errors<=eps).any():w=(errors<=eps).astype(float);w/=w.sum()
    else:w=1/np.maximum(errors,eps);w/=w.sum()
    return {'weights':dict(zip(models,w.tolist())),'calibration_MAE':dict(zip(models,errors.tolist())),
        'epsilon':eps,'calibration_cells':int(valid.sum()),'evaluation_origin':int(evaluation_origin),
        'calibration_max_label':int(frame.target_t.max()),'common_cells_only':True,
        'architecture':'inverse-calibration-MAE convex forecast combination'}

def combine(frame,configuration):
    models=list(configuration['weights']);p=frame.pivot(index=['entity','target_t'],columns='model',values='prediction')
    # No silent renormalization/fallback when a component fails.
    return sum(p[m]*configuration['weights'][m] for m in models)
