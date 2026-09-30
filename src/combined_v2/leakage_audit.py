from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
from combined_v2.data import panels,assemble,DATASETS,S
from feature_engineering.pilot_features import fit_state,origin_features,BUSINESS
from preprocessing.pilot_panels import tindex
from combined_v2.hybrid import fit,predict

def run():
    allpanels=panels();checks=[]
    for d,p in allpanels.items():
        old=pd.read_csv(ROOT/f'data/processed/pilot/{d}_pilot_cohort.csv',dtype={'entity':str})
        assert set(p.entity)==set(old.entity)
        assert not p.duplicated(['entity','t']).any()
        assert p.t.max()<tindex(d,S[d]['folds']['final_test'][0]['forecast_start'])
        assert p.loc[p.assumed_zero,'demand'].eq(0).all()
        assert (~p.loc[~p.global_recording_mask,'score_mask']).all()
        checks.append({'dataset':d,'check':'unchanged cohort, unique keys, zero/global-gap masks, final-test exclusion','status':'pass'})
        for fold in range(1,5):
            cutoff=tindex(d,S[d]['folds']['calibration'][fold-1]['origin']);state=fit_state(p,cutoff);e=sorted(state['entities'])[0]
            altered=p.copy()
            for col in ['demand']+BUSINESS:altered.loc[altered.t>cutoff,col]=9999999.
            assert state==fit_state(altered,cutoff)
            a=origin_features(p,d,e,cutoff,S[d]['horizon_days'],state);b=origin_features(altered,d,e,cutoff,S[d]['horizon_days'],state)
            for i in range(3):np.testing.assert_array_equal(a[i],b[i])
            checks.append({'dataset':d,'fold':fold,'check':'future target/business mutation invariance: lags, rolls, transforms, features','status':'pass'})
    arrays,inf,truth,states,proof=assemble(allpanels,1)
    assert np.isfinite(arrays[0]).all() and np.isfinite(arrays[1]).all()
    assert (proof.max_input_timestamp<=proof.origin).all() and (proof.max_label_timestamp<=proof.training_cutoff).all()
    assert set(truth.series_id)==set(d+'::'+e for d,p in allpanels.items() for e in p.entity.unique())
    assert len(np.unique(arrays[3]))==36 and set(arrays[3])==set(range(36))
    for d,g in proof.groupby('dataset'):assert g.origin.is_monotonic_increasing
    checks.append({'check':'pooled entity namespaces, finite inputs, chronological order and all training label timestamps','status':'pass'})
    for bad in [pd.DataFrame({'fold':[2]}),pd.DataFrame({'fold':[1],'target_t':[1],'origin':[2]})]:
        try:fit(bad,'inverse_MAE',1)
        except AssertionError:pass
        else:raise AssertionError('Hybrid accepted invalid training provenance')
    try:predict(pd.DataFrame({'fold':[1]}),{'max_fit_fold':1})
    except AssertionError:pass
    else:raise AssertionError('Hybrid accepted calibration/evaluation fold overlap')
    checks.append({'check':'Hybrid rejects future fit folds, non-out-of-sample labels and overlapping prediction folds','status':'pass'})
    proof.to_csv(ROOT/'experiments/combined_v2/training_provenance_checked.csv.gz',index=False)
    files=list((ROOT/'src/combined_v2').glob('*.py'))+[ROOT/'src/feature_engineering/pilot_features.py',ROOT/'src/preprocessing/pilot_panels.py',ROOT/'experiments/combined_v2/protocol.json',ROOT/'experiments/split_manifest.json']+list((ROOT/'data/processed/combined_v2').glob('*_panel.csv.gz'))
    report={'status':'pass','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'limits':'36-series retrospective pooled study; all four folds future-mutation tested, first-fold full training windows checked here and every fold checked at runtime; no claim that unavailable modalities exist'}
    (ROOT/'experiments/combined_v2/leakage_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(checks),'grouped leakage checks',flush=True)

if __name__=='__main__':run()
