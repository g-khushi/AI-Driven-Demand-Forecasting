"""Score frozen final forecasts once. No fitting, selection or parameter changes."""
from pathlib import Path
import sys,json,hashlib,datetime
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
from evaluation.metrics import metrics
from combined_v2.hybrid import MODELS,objective
from reporting.evaluate_combined_v2 import in_sample_blend,corrected,md
from final_experiment.common import CFG,E,R,M,guard

def run():
    guard();assert not (E/'test_scored.json').exists(),'Final test has already been scored; do not rerun selection or overwrite results'
    completed=json.loads((E/'forecast_complete.json').read_text());path=R/'test_component_predictions.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==completed['predictions_sha256']
    raw=pd.read_csv(path,dtype={'entity':str});assert set(raw.fold)=={1,2,3,4} and raw.stage.eq('final_test').all()
    assert not raw.duplicated(['dataset','entity','fold','horizon','model']).any()
    # Compute complete-cell mask across all five components and preserve coverage.
    keys=['dataset','entity','fold','target_t'];pivot=raw.pivot(index=keys,columns='model',values='prediction')
    truth=raw.drop_duplicates(keys).set_index(keys).reindex(pivot.index)
    valid=truth.score_mask&truth.y.notna()&pivot[MODELS].notna().all(axis=1)
    common=raw.set_index(keys).loc[pivot.index[valid]].reset_index()
    out=[common];own=[raw]
    for source,destination in [(common,out),(raw,own)]:
        for name,cfg in [('Hybrid_reference',CFG['reference_hybrid']),('Hybrid_v2',CFG['hybrid'])]:
            hybrid=in_sample_blend(source,cfg);hybrid['model']=name;destination.append(hybrid)
            if name=='Hybrid_v2':
                cc=CFG['corrections'][name];destination.append(corrected(hybrid,cc['offsets'],cc['factor'],name+'_corrected'))
        comparator=source[source.model.eq('Transformer')].copy();comparator['model']='MARIOD_forecasting_reconstruction';destination.append(comparator)
        cc=CFG['corrections']['MARIOD_forecasting_reconstruction'];destination.append(corrected(comparator,cc['offsets'],cc['factor'],'MARIOD_forecasting_reconstruction_corrected'))
    results=pd.concat(out,ignore_index=True);own=pd.concat(own,ignore_index=True)
    results.to_csv(R/'final_predictions_common.csv',index=False);own.to_csv(R/'final_predictions_all.csv',index=False)
    accuracy=[];coverage=[]
    for population,frame in [('common_all_components',results),('own_successful',own)]:
        for (d,model),g in frame.groupby(['dataset','model']):
            valid=g.score_mask&g.y.notna()&g.prediction.notna();z=g[valid]
            accuracy.append({'dataset':d,'model':model,'population':population,**metrics(z.y,z.prediction,z.weight if d=='favorita' else None)})
            if population=='own_successful':coverage.append({'dataset':d,'model':model,'scheduled_cells':len(g),'scorable_cells':int(g.score_mask.sum()),'successful_scorable':int(valid.sum()),'failed_scorable':int((g.score_mask&g.prediction.isna()).sum())})
    accuracy=pd.DataFrame(accuracy);accuracy.to_csv(R/'final_accuracy.csv',index=False);pd.DataFrame(coverage).to_csv(R/'failure_coverage.csv',index=False)
    main=accuracy[accuracy.population.eq('common_all_components')]
    combined=pd.DataFrame([{'model':m,'equal_dataset_normalized_MAE':objective(g)} for m,g in results.groupby('model')]);combined.to_csv(R/'combined_accuracy.csv',index=False)
    hname='Hybrid_v2_corrected';bname='MARIOD_forecasting_reconstruction_corrected'
    comparisons=[]
    pairs=[(bname,hname),('MARIOD_forecasting_reconstruction','Hybrid_v2'),('Hybrid_v2',hname),('MARIOD_forecasting_reconstruction',bname),('Hybrid_reference',hname)]+[(m,hname) for m in MODELS]
    for d,g in main.groupby('dataset'):
        a=g.set_index('model')
        for baseline,proposed in pairs:
            for metric in ['MAE','RMSE','sMAPE','WAPE']+(['NWRMSLE'] if d=='favorita' else []):
                b=a.loc[baseline,metric];h=a.loc[proposed,metric]
                comparisons.append({'dataset':d,'baseline':baseline,'comparison':proposed,'metric':metric,'baseline_error':b,'comparison_error':h,'absolute_improvement':b-h,'percentage_improvement':100*(b-h)/b if b else None})
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(R/'accuracy_and_correction_comparisons.csv',index=False)
    segmented=[]
    for dimension in ['fold','horizon','regime','volume','entity']:
        for (d,model,value),g in results.groupby(['dataset','model',dimension]):segmented.append({'dataset':d,'model':model,'dimension':dimension,'segment':value,**metrics(g.y,g.prediction,g.weight if d=='favorita' else None)})
    segments=pd.DataFrame(segmented);segments.to_csv(R/'error_analysis.csv',index=False)
    consistency=[]
    for d in main.dataset.unique():
        for dim in ['fold','horizon','entity']:
            a=segments[segments.dataset.eq(d)&segments.dimension.eq(dim)].pivot(index='segment',columns='model',values='MAE');delta=a[bname]-a[hname]
            consistency.append({'dataset':d,'dimension':dim,'Hybrid_lower_MAE':int((delta>1e-10).sum()),'Hybrid_higher_MAE':int((delta< -1e-10).sum()),'ties':int((delta.abs()<=1e-10).sum()),'total':len(delta)})
    consistency=pd.DataFrame(consistency);consistency.to_csv(R/'comparison_consistency.csv',index=False)
    resource=json.loads((E/'full_population_feasibility.json').read_text())
    n=combined.set_index('model').equal_dataset_normalized_MAE;improvement=float(100*(n[bname]-n[hname])/n[bname])
    per_dataset=comparisons[comparisons.baseline.eq(bname)&comparisons.comparison.eq(hname)&comparisons.metric.eq('MAE')]
    verdict={'primary_metric':'equal-dataset normalized MAE','Hybrid':float(n[hname]),'forecasting_only_comparator':float(n[bname]),'Hybrid_error_reduction_percent':improvement,'Hybrid_better_on_primary_metric':bool(improvement>0),'all_datasets_MAE_improved':bool((per_dataset.absolute_improvement>0).all()),'comparison_scope':'Corrected Hybrid vs corrected compact forecasting-only reconstruction. Not full MARIOD. 36 fixed series and four original final-test origins per source.','full_population_completed':False,'full_population_blocker':f"Current feature arrays alone approximately {resource['feature_arrays_alone_GiB']:.1f} GiB versus 8 GiB RAM; streaming implementation and additional resources needed.",'statistical_significance_established':False,'post_test_model_changes':False}
    (R/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    primary=main[main.model.isin([hname,bname])].copy();primary['model']=primary.model.map({hname:'Corrected Hybrid',bname:'Corrected MARIOD-style forecaster'})
    corrections=comparisons[((comparisons.baseline=='Hybrid_v2')&(comparisons.comparison==hname))|((comparisons.baseline=='MARIOD_forecasting_reconstruction')&(comparisons.comparison==bname))]
    compute=pd.read_csv(R/'fold_1_compute.csv');inference=pd.read_csv(R/'test_inference_costs.csv');total_fit=float(compute.training_seconds.sum());total_inference=float(inference.inference_seconds.sum())
    lines=['# Frozen final-test comparison','',
        f"**Primary result:** corrected Hybrid has {abs(improvement):.2f}% {'lower' if improvement>0 else 'higher'} equal-dataset normalized MAE than the corrected compact MARIOD-style forecasting reconstruction ({n[hname]:.6f} versus {n[bname]:.6f}). This does not establish superiority over full MARIOD or over the complete dataset populations.",'',
        '## What was completed','',
        'Forecasting-only comparator and its residual correction; configuration/checksum freeze before test preprocessing; pre-test leakage audit; one refit of all five components using pre-test observations; forecasts across all four original final-test origins with fixed parameters; common-population accuracy, before/after correction, failure coverage, and horizon/entity/origin analysis. Both methods have the same three-factor correction-selection budget on calibration folds 1–2. No correction or architecture was chosen using final-test errors.','',
        'This remains the original 36-series cohort: 12 per dataset selected from initial-training information. Full-population training was not performed. Final test dates: Dunnhumby DAY 656–711, Favorita 2017-06-13–2017-08-15, UCI 2011-10-14–2011-12-08. Each source retains its original time axis.','',
        '## Primary accuracy comparison','',md(primary,['dataset','model','n','MAE','RMSE','MAPE','MAPE_coverage','sMAPE','WAPE','Bias','NWRMSLE']),'',
        'Per-dataset MAE gains (positive favors Hybrid; negative is deterioration):','',md(per_dataset,['dataset','baseline_error','comparison_error','percentage_improvement']),'',
        '## Error correction before and after','',md(corrections[corrections.metric.eq('MAE')],['dataset','baseline','baseline_error','comparison_error','percentage_improvement']),'',
        'Corrections add a frozen dataset-specific median normalized residual multiplied by the fixed pre-test scale and selected shrinkage, then clip to nonnegative demand. The paper does not supply this residual-correction method: it is an explicitly shared extension. Correction may improve one metric while worsening another. Full MAE/RMSE/sMAPE/WAPE and Favorita NWRMSLE comparisons are in `accuracy_and_correction_comparisons.csv`.','',
        '## All model scores','',md(main,['dataset','model','MAE','RMSE','sMAPE','WAPE','NWRMSLE']),'',
        'Combined normalized scores (mean of each dataset’s mean absolute error divided by each series’s frozen pre-test mean scale, floored at one):','',md(combined,['model','equal_dataset_normalized_MAE']),'',
        'Raw demand units are not summed across unrelated retailers. MAPE excludes zero actuals, sMAPE assigns zero to both-zero cells, and Favorita NWRMSLE uses perishable weights 1.25 versus 1. Primary scores use the exact same successful/scorable cells for all five components; own-successful metrics and failure counts are also saved. No missing forecast is silently replaced.','',
        '## Consistency and statistical limits','',md(consistency,list(consistency.columns)),'',
        'Only four forecast-origin blocks per source are available. Reliable statistical significance or full-population superiority is not established; individual rows and products are not treated as independent trials. No p-values are manufactured. The direction of error improvement must be read by dataset, metric and origin.','',
        '## Frozen-model integrity','',
        'Neural/tree parameters, transforms, Hybrid weights and residual offsets stay fixed throughout testing. Later origins may use observations already available by that origin. SARIMA filtering state is refreshed with fixed coefficients (`refit=False`); coefficients are checked for exact equality. Saved-model replay reproduces first-origin forecasts within 1e-6, and all artifact checksums remain unchanged.','',
        'The comparator is exactly the Hybrid’s compact cross-modal Transformer component, with its own separately calibrated correction. It uses the same data and forecasting features; it is not an independent full-MARIOD implementation. The paper’s hierarchical inventory agents, sensor streams and inventory optimization are unavailable. Published paper metrics use a different unspecified scoring population and are not ranked against these results.','',
        '## Resources and unfinished full-scale work','',
        f'Final component fitting took {total_fit:.2f} seconds in total (excluding feature preparation and serialization); summed component forecasting/filtering time across all test origins was {total_inference:.3f} seconds. Hybrid deployment needs all five components; the forecasting comparator needs only Transformer. Calibration/tuning costs are additional. Component timings and memory are recorded in `fold_1_compute.csv` and `test_inference_costs.csv`.', '',
        f"The training-only eligibility lists contain {sum(x['eligible_entities'] for x in resource['datasets']):,} entities. The current full-population feature materialization would require approximately {resource['feature_arrays_alone_GiB']:.1f} GiB for feature arrays alone, before models and copies, versus this machine’s 8 GiB RAM and roughly {resource['free_disk_GiB']:.1f} GiB free disk at assessment. No unsafe full-population run was launched. It needs an out-of-core implementation and additional compute/storage; this is a real unfinished task, not completed by subsampling.",'',
        'Full MARIOD additionally needs the authors’ implementation or fully specified reconstruction plus real inventory/sensor/graph/cost inputs. Those inputs have not been supplied. Consequently the answer to “is my Hybrid better than full MARIOD?” remains unproven, regardless of this forecasting-only result.','',
        'All prior experiments are preserved. Final-test results are final evidence for these frozen choices; they must not be used to retune and report this same test as untouched again. See `experiments/final_experiment/README.md` for artifact locations and reproduction rules.','']
    (R/'final_comparison_report.md').write_text('\n'.join(lines))
    (E/'test_scored.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'predictions_sha256':completed['predictions_sha256'],'configuration_sha256':hashlib.sha256((E/'configuration_freeze.json').read_bytes()).hexdigest(),'verdict':verdict,'scored_once':True},indent=2)+'\n')
    print(json.dumps(verdict,indent=2))

if __name__=='__main__':run()
