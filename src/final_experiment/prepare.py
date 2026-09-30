"""Materialize final-test panels only after configuration freeze; preserve cohorts."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import pandas as pd
from preprocessing import pilot_panels as source

def run(dataset):
    assert (ROOT/'experiments/final_experiment/configuration_freeze.json').exists()
    out=ROOT/'data/processed/final_experiment';source.OUT=out
    source.bounds=lambda d:(source.tindex(d,source.SPLIT[d]['initial_training_end']),source.tindex(d,source.SPLIT[d]['folds']['final_test'][-1]['forecast_end']))
    getattr(source,dataset)()
    old=pd.read_csv(ROOT/f'data/processed/combined_v2/{dataset}_panel.csv.gz',dtype={'entity':str,'store':str})
    new=pd.read_csv(out/f'{dataset}_panel.csv.gz',dtype={'entity':str,'store':str})
    cols=['entity','t','demand','promotion','price','transactions','customers','oil','holiday','weight','score_mask']
    a=old[cols].sort_values(['entity','t']).reset_index(drop=True)
    b=new[new.t<=old.t.max()][cols].sort_values(['entity','t']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-12)
    assert set(old.entity)==set(new.entity)
    ledger_path=out/f'{dataset}_preprocessing_ledger.json';ledger=json.loads(ledger_path.read_text())
    ledger['scope']='36-series final experiment: 12 fixed entities for this dataset; historical observations unchanged'
    ledger['final_test_access']='Test panel materialized after configuration freeze. Test labels excluded from fitting, transforms and calibration; used only for scoring and later-origin observed context.'
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(dataset,'historical panel and cohort preserved; final-test rows appended',flush=True)

if __name__=='__main__':run(sys.argv[1])
