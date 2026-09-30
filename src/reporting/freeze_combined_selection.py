"""Freeze choices using folds 1–2 before generating later evaluation forecasts."""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import pandas as pd
from reporting.evaluate_combined_v2 import on_common,in_sample_blend,residual_offsets,corrected
from combined_v2.hybrid import CANDIDATES,fit,predict,objective

def run():
    target=ROOT/'experiments/combined_v2/selection_freeze.json'
    if target.exists():raise RuntimeError('Selection freeze already exists. Archive the entire experiment before a new run; do not overwrite after evaluation.')
    assert not any((ROOT/f'results/combined_v2/fold_{i}_component_predictions.csv').exists() for i in [3,4]),'Later evaluation forecasts already exist'
    paths=[ROOT/f'results/combined_v2/fold_{i}_component_predictions.csv' for i in [1,2]]
    one,two=[on_common(pd.read_csv(p,dtype={'entity':str})) for p in paths]
    trials=[];configs={}
    for method in CANDIDATES:
        cfg=fit(one,method,1);configs[method]=cfg;trials.append({'method':method,'score':objective(predict(two,cfg))})
    best=min(trials,key=lambda r:r['score'])['method'];cfg=fit(pd.concat([one,two]),best,2)
    offsets=residual_offsets(in_sample_blend(one,configs[best]));b=predict(two,configs[best])
    corrections=[{'factor':factor,'score':objective(corrected(b,offsets,factor,'Hybrid_v2_corrected'))} for factor in [0.,.5,1.]]
    choice=min(corrections,key=lambda r:r['score'])['factor']
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_predictions_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'selection_only_folds':[1,2],'hybrid_trials':trials,'selected_method':best,'frozen_hybrid':cfg,'correction_trials':corrections,'selected_correction_factor':choice,'frozen_correction_offsets':residual_offsets(in_sample_blend(pd.concat([one,two]),cfg)),'later_predictions_present_at_freeze':[False,False]}
    target.write_text(json.dumps(result,indent=2)+'\n');print('Frozen:',best,'correction factor',choice)

if __name__=='__main__':run()
