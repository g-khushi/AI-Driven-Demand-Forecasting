"""Bounded method/correction selection; later calibration evaluation only."""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
from combined_v2.hybrid import MODELS,CANDIDATES,fit,predict,objective,examples
from evaluation.metrics import metrics
R=ROOT/'results/combined_v2';E=ROOT/'experiments/combined_v2';M=ROOT/'models/combined_v2'

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def score(g):return metrics(g.y,g.prediction,g.weight if g.dataset.iloc[0]=='favorita' else None)
def md(frame,cols):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}'
        return str(v)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame[cols].itertuples(index=False,name=None))

def on_common(frame):
    x,y=examples(frame);return frame.set_index(['dataset','entity','fold','target_t']).loc[x.index].reset_index()

def calibrated_prediction(frame,cfg,model_name):
    out=predict(frame,cfg);out['model']=model_name;return out

def in_sample_blend(frame,cfg):
    # For correction fitting only. Never reported as evaluation accuracy.
    keys=['dataset','entity','fold','target_t'];x=frame.pivot(index=keys,columns='model',values='prediction').reindex(columns=MODELS)
    y=frame.drop_duplicates(keys).set_index(keys).reindex(x.index).copy();values=[]
    for key,row in y.iterrows():
        d=key[0];method=cfg['method']
        w=cfg['pooled_weights'] if method=='pooled_MAE' else cfg['groups'].get(d+'::'+row.regime,cfg['groups'][d]) if method=='regime_shrinkage' else cfg['groups'][d]
        values.append(float(x.loc[key].to_numpy()@np.asarray(w)))
    y['prediction']=values;return y.reset_index()

def residual_offsets(frame):
    f=frame[frame.score_mask&frame.y.notna()&frame.prediction.notna()].copy();f['normalized_residual']=(f.y-f.prediction)/f.scale
    return f.groupby('dataset').normalized_residual.median().to_dict()

def corrected(frame,offsets,factor,name):
    f=frame.copy();f['prediction']=(f.prediction+f.scale*f.dataset.map(offsets)*factor).clip(lower=0);f['model']=name;return f

def run():
    frames=[pd.read_csv(R/f'fold_{i}_component_predictions.csv',dtype={'entity':str}) for i in range(1,5)]
    allf=pd.concat(frames,ignore_index=True);assert set(allf.model)==set(MODELS)
    # Same complete and scorable cells for all methods, with explicit coverage table.
    coverage=[]
    for (d,fold,model),g in allf.groupby(['dataset','fold','model']):coverage.append({'dataset':d,'fold':fold,'model':model,'scheduled':len(g),'scorable':int(g.score_mask.sum()),'successful_scorable':int((g.score_mask&g.prediction.notna()).sum())})
    pd.DataFrame(coverage).to_csv(R/'coverage.csv',index=False)
    f=[on_common(a) for a in frames];one,two,three,four=f;development=pd.concat([one,two]);evaluation=pd.concat([three,four])
    trials=[];candidate_configs={}
    for method in CANDIDATES:
        cfg=fit(one,method,1);forecast=predict(two,cfg);loss=objective(forecast)
        trials.append({'candidate':method,'fold_2_equal_domain_normalized_MAE':loss,'fit_folds':'1','selection_fold':2});candidate_configs[method]=cfg
    winner=min(trials,key=lambda x:x['fold_2_equal_domain_normalized_MAE'])['candidate']
    pd.DataFrame(trials).to_csv(R/'hybrid_selection_trials.csv',index=False)
    selected=fit(development,winner,2);reference=fit(development,'inverse_MAE',2)
    frozen=json.loads((E/'selection_freeze.json').read_text())
    assert frozen['selected_method']==winner and frozen['frozen_hybrid']==selected
    assert frozen['later_predictions_present_at_freeze']==[False,False]
    for p,h in frozen['training_predictions_sha256'].items():
        source=Path(p)
        if not source.exists():source=ROOT/'results/combined_v2'/source.name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==h
    dump(M/'selected_hybrid.json',selected);dump(M/'reference_hybrid.json',reference)
    reference_predictions=calibrated_prediction(evaluation,reference,'Hybrid_reference')
    proposed=calibrated_prediction(evaluation,selected,'Hybrid_v2')
    baseline_one=in_sample_blend(one,candidate_configs[winner]);baseline_two=predict(two,candidate_configs[winner])
    sources=[('Hybrid_v2',baseline_one,baseline_two,in_sample_blend(development,selected),proposed)]
    approval=E/'comparator_authorization.json';comparator_enabled=approval.exists() and json.loads(approval.read_text()).get('forecasting_only_authorized',False)
    if comparator_enabled:
        # Same shared forecasting component, same cohort and compute budget.
        def transformer(g):
            out=g[g.model.eq('Transformer')].copy();out['model']='MARIOD_forecasting_reconstruction';return out
        sources.append(('MARIOD_forecasting_reconstruction',transformer(one),transformer(two),transformer(development),transformer(evaluation)))
    correction_trials=[];correction_choices={};forecast_frames=[evaluation,reference_predictions,proposed]
    if comparator_enabled:forecast_frames.append(sources[1][4])
    for name,a,b,dev,test in sources:
        offsets=residual_offsets(a);options=[]
        for factor in [0.,.5,1.]:
            v=corrected(b,offsets,factor,name+'_corrected');obj=objective(v)
            row={'model':name,'factor':factor,'fit_fold':1,'selection_fold':2,'selection_normalized_MAE':obj};options.append(row);correction_trials.append(row)
        best=min(options,key=lambda r:r['selection_normalized_MAE'])['factor'];final_offsets=residual_offsets(dev)
        if name=='Hybrid_v2':
            assert best==frozen['selected_correction_factor']
            assert final_offsets==frozen['frozen_correction_offsets']
        correction_choices[name]={'factor':best,'offsets':final_offsets,'fit_max_fold':2,'selection':'fold 2 only; ties prefer smaller correction','correction':'add scale times dataset median normalized residual times selected shrinkage, then clip at zero'}
        forecast_frames.append(corrected(test,final_offsets,best,name+'_corrected'))
    dump(M/'correction_configuration.json',correction_choices);pd.DataFrame(correction_trials).to_csv(R/'correction_selection_trials.csv',index=False)
    forecast=pd.concat(forecast_frames,ignore_index=True);forecast.to_csv(R/'heldout_development_predictions.csv',index=False)
    accuracy=[]
    for (d,name),g in forecast.groupby(['dataset','model']):accuracy.append({'dataset':d,'model':name,'evaluation':'calibration folds 3–4 only',**score(g)})
    acc=pd.DataFrame(accuracy);acc.to_csv(R/'accuracy.csv',index=False)
    normalized=[]
    for name,g in forecast.groupby('model'):normalized.append({'model':name,'equal_dataset_mean_normalized_MAE':objective(g),'evaluated_fold_count':2})
    norm=pd.DataFrame(normalized);norm.to_csv(R/'combined_normalized_accuracy.csv',index=False)
    gains=[]
    for d in ['dunnhumby','favorita','uci']:
        a=acc[acc.dataset.eq(d)].set_index('model')
        pairs=[('Hybrid_reference','Hybrid_v2'),('Hybrid_v2','Hybrid_v2_corrected')]
        if comparator_enabled:pairs += [('MARIOD_forecasting_reconstruction','MARIOD_forecasting_reconstruction_corrected'),('MARIOD_forecasting_reconstruction','Hybrid_v2'),('MARIOD_forecasting_reconstruction_corrected','Hybrid_v2_corrected')]
        for baseline,proposed_name in pairs:
            for metric in ['MAE','RMSE','sMAPE','WAPE']+(['NWRMSLE'] if d=='favorita' else []):
                b=a.loc[baseline,metric];h=a.loc[proposed_name,metric]
                gains.append({'dataset':d,'baseline':baseline,'comparison':proposed_name,'metric':metric,'baseline_error':b,'comparison_error':h,'absolute_improvement':b-h,'percentage_improvement':100*(b-h)/b if b else None})
    gains=pd.DataFrame(gains);gains.to_csv(R/'improvements_and_error_correction.csv',index=False)
    segments=[]
    for dimension in ['horizon','regime','volume','fold']:
        for (d,name,segment),g in forecast.groupby(['dataset','model',dimension]):segments.append({'dataset':d,'model':name,'dimension':dimension,'segment':segment,**score(g)})
    pd.DataFrame(segments).to_csv(R/'error_segments.csv',index=False)
    costs=pd.concat([pd.read_csv(R/f'fold_{i}_compute.csv') for i in range(1,5)]);costs.to_csv(R/'compute.csv',index=False)
    status={'status':'combined_hybrid_development_complete','comparator_enabled':comparator_enabled,'full_MARIOD_reproduced':False,'final_test_used':False,'hybrid_selected_method':winner,'correction_choices':correction_choices,'scope':'36 fixed series, two later calibration evaluation origins; not full-population or final-test results'}
    dump(E/'result_status.json',status)
    files=list((ROOT/'src/combined_v2').glob('*.py'))+[Path(__file__),E/'protocol.json',ROOT/'experiments/split_manifest.json']+list(M.glob('*.json'))
    dump(E/'development_snapshot.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Reproducible development snapshot; not a final-test freeze','checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    ntable=norm.set_index('model');overall=100*(ntable.loc['Hybrid_reference','equal_dataset_mean_normalized_MAE']-ntable.loc['Hybrid_v2_corrected','equal_dataset_mean_normalized_MAE'])/ntable.loc['Hybrid_reference','equal_dataset_mean_normalized_MAE']
    measured=[]
    for d in ['dunnhumby','favorita','uci']:
        a=acc[acc.dataset.eq(d)].set_index('model');base=a.loc['Hybrid_reference','MAE'];new=a.loc['Hybrid_v2_corrected','MAE'];measured.append(f'{d}: {100*(base-new)/base:.2f}% MAE improvement')
    lines=['# Combined-dataset Hybrid development experiment','',
        'Three datasets are combined as a retailer-labelled panel of the unchanged 36-series cohort (12 from each source). All raw sources and the earlier pilot are preserved. This is not full-population training. The development panel contains 23,318 daily rows; it ends before each original final-test boundary.','',
        f'Corrected Hybrid versus the inverse-MAE reference on later development folds: {overall:.2f}% improvement in equal-dataset normalized MAE. '+ '; '.join(measured)+'. Negative improvements are deterioration. MAE and RMSE trade-offs must be read together; these results do not establish universal superiority. No evaluation-based corrections were made after seeing them.','', '## What changed','',
        'XGBoost, LSTM and Transformer now share training across the three sources with distinct entity IDs and domain indicators. Targets are scaled by training-only entity means, and training weights balance the three datasets. Naive and SARIMA remain local time-series models. Native-unit errors are reported separately; the combined score averages normalized MAE equally across datasets. Different historical calendars are an offline multi-domain experiment, not a claim of contemporaneous operational forecasts.','',
        'The Transformer component follows the paper’s pre-normalized temporal attention and modality-specific cross-attention idea (equations 9–12), using history, recorded promotion/price, and available external/business features. Its compact 64-dimensional, four-head architecture is an explicit CPU adaptation. Missing inventory, sensors and competitors are not fabricated. This is not the paper’s full 512-dimensional, six-layer forecasting-plus-RL architecture.','',
        'Hybrid tries four declared weighting rules: inverse MAE, shared normalized-MAE optimization, dataset-specific shrinkage, and intermittency-regime shrinkage. Optimized convex weights retain at least 1% for each of the five components. The two shrinkage constants are fixed, not searched. All trials and any deterioration are retained.','',
        '## Selection and error correction','',
        'Fold 1 supplies out-of-sample errors. Fold 2 selects the weighting rule and one correction shrinkage factor from 0, 0.5 or 1. Weights and residual offsets are then fitted using folds 1–2 only and frozen for folds 3–4. Components are refitted at each later origin using only then-available history; Hybrid parameters remain fixed. No changes are selected from later evaluation errors. The pre-existing final test is unused.','',
        f'Selected Hybrid method: **{winner}**. The objective is equal-dataset mean normalized MAE, not whichever metric later looks best. Error correction adds a dataset median normalized residual, scaled back into native units; negative forecasts are clipped to zero. This is our declared extension, not a residual-correction algorithm supplied by the paper.','',md(pd.DataFrame(trials),['candidate','fold_2_equal_domain_normalized_MAE']),'',
        '## Measured accuracy on later calibration folds','',md(acc,['dataset','model','n','MAE','RMSE','MAPE','MAPE_coverage','sMAPE','WAPE','Bias','NWRMSLE']),'',
        'MAPE excludes zero actuals and reports coverage; sMAPE treats both-zero cells as zero error. Favorita NWRMSLE uses the original 1/1.25 item weights. Scoring uses identical successful/scorable cells for all compared components and Hybrids; failures and scheduled counts are saved separately.','',
        'Combined normalized scores (smaller is better):','',md(norm,['model','equal_dataset_mean_normalized_MAE']),'',
        'The reference Hybrid uses the earlier inverse-MAE rule with the same newly fitted components. This isolates the weighting change; it is not a direct comparison with the earlier pilot’s different models and dates.','',
        '## Accuracy improvements and correction effects','',md(gains[gains.metric.eq('MAE')],['dataset','baseline','comparison','baseline_error','comparison_error','percentage_improvement']),'',
        'Positive improvements favor the comparison method; negative values are deterioration. All metrics and every correction trial are in the accompanying CSVs. A zero selected factor means correction was rejected during development. RMSE and MAE can favor different choices.','',
        '## MARIOD comparison boundary','']
    if comparator_enabled:
        lines += ['The user authorized a forecasting-only reconstruction. It is the same cross-modal Transformer component trained on the same combined data, with the same feature availability and epochs. Its correction has the same three-factor budget and chronological selection rule as Hybrid. This is an approximate, compact forecasting comparator, **not full MARIOD**. Improvements over it cannot be stated as improvements over the published full inventory/RL system.','']
    else:
        lines += ['**MARIOD accuracy and correction comparison is pending the user’s reproduction-scope choice.** The supplied data lack inventories, sensors, supply-chain topology and operating costs required by full MARIOD. The user was asked whether to authorize a clearly labelled forecasting-only reconstruction or supply the authors’ implementation and additional data. No made-up MARIOD predictions or accuracy numbers are reported.','']
    lines += ['The paper’s Table 2 numbers are not comparable with this cohort and evaluation: they do not identify dataset-specific prediction rows, horizons or scoring population. They are not used as target scores to beat. See `paper_reproduction_assessment.md` for the exact omissions and architecture mapping.','',
        '## Limitations and reproducibility','',
        'Only two later evaluation origins per source are available, so reliable statistical superiority or full-population generalization is not established. Sparse demand, conditional zero filling, short history and unavailable business modalities remain substantive limitations. No p-values are manufactured. The experiment uses actual observed purchase/sales targets, not reconstructed latent demand or synthetic inventories.','',
        'A full-population run and full MARIOD reproduction remain separate, unfinished work. The paper reports an eight-A100 training setup; this is a bounded CPU experiment. Full inventory error/stockout comparisons are impossible from the available data alone.','',
        'Code: `src/combined_v2/`; protocol, data manifest and leakage gate: `experiments/combined_v2/`; model states: `models/combined_v2/`; all folds, metrics, correction trials, segments and measured resource costs: `results/combined_v2/`. The previous pilot remains unchanged.','']
    (R/'combined_hybrid_report.md').write_text('\n'.join(lines));print(json.dumps(status,indent=2))

if __name__=='__main__':run()
