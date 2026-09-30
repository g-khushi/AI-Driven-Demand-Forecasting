"""Reuse the unchanged component trainer at the one pre-test fitting cutoff."""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
from final_experiment.common import panels,assemble_training,guard,R,M,E
from combined_v2 import run_components as runner

def run():
    guard()
    assert not (E/'training_complete.json').exists(),'Final parameters already fitted; refusing retraining'
    assert not (E/'test_scored.json').exists(),'Final test already scored; no retraining allowed'
    runner.OUT=R;runner.ART=M;runner.panels=panels;runner.assemble=assemble_training;runner.guard=guard
    runner.run(1)
    files=list(M.glob('*'))
    (E/'training_complete.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parameter_updates_after_this_point':False,'model_checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')

if __name__=='__main__':run()
