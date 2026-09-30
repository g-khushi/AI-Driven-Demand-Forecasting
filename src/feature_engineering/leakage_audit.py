"""Fail-closed leakage tests on real pilot panels, including future mutation tests.
Mutated copies are test fixtures only; never saved as research data/results.
"""
from pathlib import Path
import hashlib,json,sys,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from feature_engineering.pilot_features import load_panel,fit_state,origin_features,training_arrays,BUSINESS
from preprocessing.pilot_panels import tindex,stats_from_lines
from hybrid.hybrid_model import assert_calibration_boundary
S=json.loads((ROOT/'experiments/split_manifest.json').read_text())['datasets']

def checksum(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    tests=[];provenance=[]
    def check(dataset,name,fn):
        try:
            fn();tests.append({'dataset':dataset,'test':name,'status':'pass'})
        except Exception as e:tests.append({'dataset':dataset,'test':name,'status':'fail','error':repr(e)})
    for dataset in S:
        panel=load_panel(dataset);origin=tindex(dataset,S[dataset]['folds']['calibration'][0]['origin']);H=S[dataset]['horizon_days']
        st=fit_state(panel,origin);entity=sorted(st['entities'])[0]
        ref=origin_features(panel,dataset,entity,origin,H,st)
        for label,columns in [('lag_leakage',['demand']),('rolling_window_leakage',['demand']),
            ('future_promotion_leakage',['promotion']),('future_oil_price_leakage',['oil']),
            ('future_transaction_count_leakage',['transactions']),('future_customer_count_leakage',['customers']),
            ('future_holiday_leakage',['holiday']),('future_price_leakage',['price'])]:
            def mutation(columns=columns):
                altered=panel.copy()
                for col in columns:altered.loc[altered.t>origin,col]=1234567.89
                got=origin_features(altered,dataset,entity,origin,H,st)
                for i in range(3):np.testing.assert_array_equal(ref[i],got[i])
            check(dataset,label,mutation)
        def state_test():
            altered=panel.copy()
            for c in ['demand']+BUSINESS:altered.loc[altered.t>origin,c]=9876543.21
            assert st==fit_state(altered,origin)
        check(dataset,'scaler_leakage',state_test);check(dataset,'imputation_leakage',state_test)
        def encoding():
            altered=panel.copy();altered.loc[altered.t>origin,'entity']='future_unseen_entity'
            assert st==fit_state(altered,origin)
            assert st['encoding']=='entity ID only; no target encoding'
        check(dataset,'target_leakage_through_encodings',encoding)
        def selection():
            cut=tindex(dataset,S[dataset]['initial_training_end'])
            before=stats_from_lines(panel,cut)
            altered=panel.copy();altered.loc[altered.t>cut,'demand']=99999999.
            pd.testing.assert_frame_equal(before,stats_from_lines(altered,cut))
            cohort=pd.read_csv(ROOT/f'data/processed/pilot/{dataset}_cohort_eligibility.csv.gz')
            assert cohort.selection_cutoff.eq(cut).all()
            assert cohort.loc[cohort.pilot_selected,'eligible'].all()
        check(dataset,'entity_selection_leakage',selection)
        def window_check():
            _,proof=training_arrays(panel,dataset,origin,H,st)
            assert (proof.max_input_timestamp<=proof.origin).all()
            assert (proof.max_label_timestamp<=proof.training_cutoff).all()
            proof['dataset']=dataset;provenance.append(proof)
        check(dataset,'input_timestamp_and_training_label_cutoff',window_check)
        def final_check():
            start=tindex(dataset,S[dataset]['folds']['final_test'][0]['forecast_start'])
            assert panel.t.max()<start
            assert not panel.duplicated(['entity','t']).any()
            assert panel.loc[panel.assumed_zero,'demand'].eq(0).all()
            assert (~panel.loc[panel.assumed_zero,'observation_mask']).all()
            assert panel.loc[~panel.global_recording_mask,'score_mask'].eq(False).all()
            assert panel.loc[panel.demand.isna(),'score_mask'].eq(False).all()
        check(dataset,'test_exclusion_and_observation_masks',final_check)
    def hybrid_guard():
        # Contract-only provenance fixture; no numerical predictions or weights produced.
        good=pd.DataFrame({'stage':['calibration_fit'],'origin':[1],'target_t':[2]})
        assert_calibration_boundary(good,3)
        for bad in [good.assign(target_t=4),good.assign(stage='final_test'),good.assign(origin=3)]:
            try:assert_calibration_boundary(bad,3)
            except ValueError:continue
            raise AssertionError('Hybrid boundary accepted forbidden provenance')
    check('all','hybrid_weight_leakage',hybrid_guard)
    if provenance:pd.concat(provenance).to_csv(ROOT/'experiments/training_window_provenance.csv.gz',index=False)
    tracked=list((ROOT/'data/processed/pilot').glob('*_panel.csv.gz'))+list((ROOT/'src/feature_engineering').glob('*.py'))+list((ROOT/'src/preprocessing').glob('*.py'))+list((ROOT/'src/hybrid').glob('*.py'))+list((ROOT/'src/evaluation').glob('*.py'))+[ROOT/'experiments/continuation_protocol.json',ROOT/'experiments/split_manifest.json']
    report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'pass' if all(t['status']=='pass' for t in tests) else 'fail',
        'scope':'Pilot: 12 training-stratified entities per dataset, fixed first two calibration folds; no final test',
        'tests':tests,'checksums':{str(p.relative_to(ROOT)):checksum(p) for p in tracked},
        'limits':['No proof of original metadata publication times; business features conservatively lagged.',
          'Training-fold scalers fit only through training cutoff. Raw sample features are bounded by sample origin.',
          'Tests establish contracts on selected pilot panels, not an assertion that future code changes are safe.']}
    (ROOT/'experiments/leakage_audit_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],len(tests),'checks',flush=True)
    if report['status']!='pass':raise SystemExit(1)

if __name__=='__main__':run()
