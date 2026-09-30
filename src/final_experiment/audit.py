from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
from final_experiment.common import panels,assemble_training,inference,check_freeze,CFG,E,S,DATASETS
from feature_engineering.pilot_features import fit_state,BUSINESS
from preprocessing.pilot_panels import tindex

def run():
    check_freeze();p=panels();checks=[];states={}
    for d in DATASETS:
        cut=tindex(d,S[d]['folds']['final_test'][0]['origin']);states[d]=fit_state(p[d],cut)
        changed=p[d].copy()
        for col in ['demand']+BUSINESS:changed.loc[changed.t>cut,col]=12345678.
        assert states[d]==fit_state(changed,cut)
        assert CFG['hybrid']['diagnostics'][d]['max_label']<=cut
        assert not p[d].duplicated(['entity','t']).any()
        checks.append({'dataset':d,'test':'all test targets/covariates cannot alter training scaler, imputation, entity encoding; weights precede test','status':'pass'})
    for fold in range(1,5):
        ref,_,_=inference(p,states,fold);changed={d:g.copy() for d,g in p.items()}
        for d in DATASETS:
            origin=tindex(d,S[d]['folds']['final_test'][fold-1]['origin'])
            for col in ['demand']+BUSINESS:changed[d].loc[changed[d].t>origin,col]=7654321.
        got,_,_=inference(changed,states,fold)
        for a,b in zip(ref,got):np.testing.assert_array_equal(a,b)
        checks.append({'fold':fold,'test':'all 36 entities: future target/business mutations cannot alter any forecasting input','status':'pass'})
    arrays,inf,truth,states,proof=assemble_training(p)
    assert np.isfinite(arrays[0]).all() and np.isfinite(arrays[1]).all()
    assert (proof.max_label_timestamp<=proof.training_cutoff).all() and (proof.max_input_timestamp<=proof.origin).all()
    for d,g in proof.groupby('dataset'):assert g.training_cutoff.eq(tindex(d,S[d]['folds']['final_test'][0]['origin'])).all()
    proof.to_csv(E/'training_provenance.csv.gz',index=False)
    checks.append({'test':'every training window label ends before test; input timestamp bounded; no random shuffle','status':'pass'})
    tracked=list((ROOT/'src/final_experiment').glob('*.py'))+list((ROOT/'src/combined_v2').glob('*.py'))+[ROOT/'src/feature_engineering/pilot_features.py',ROOT/'src/preprocessing/pilot_panels.py',E/'configuration_freeze.json']+list((ROOT/'data/processed/final_experiment').glob('*_panel.csv.gz'))
    report={'status':'pass','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'checksums':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in tracked}}
    (E/'leakage_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(checks),'final-test leakage check groups',flush=True)

if __name__=='__main__':run()
