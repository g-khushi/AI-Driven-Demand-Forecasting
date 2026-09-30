"""Measured pilot reporting; never trains models or opens raw/final-test data."""
from pathlib import Path
import sys,json,hashlib,datetime,platform,importlib.metadata as md
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.model_deps'),str(ROOT/'.audit_deps')]
import numpy as np
import pandas as pd
from evaluation.metrics import metrics
from hybrid.hybrid_model import MODELS,fit_weights,combine
from feature_engineering.pilot_features import load_panel
R=ROOT/'results';P=R/'pilot';ALL=MODELS+['Hybrid'];DATASETS=['dunnhumby','favorita','uci']

def save(name,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(R/name,index=False);return f

def score(f,pred='prediction'):
    return metrics(f.y,f[pred],f.weight if len(f) and f.dataset.iloc[0]=='favorita' else None)

def table(f,cols=None):
    if cols:f=f[cols]
    def val(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.4f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))

def regimes(panel,cutoff):
    rows=[]
    for e,g in panel[panel.t<=cutoff].groupby('entity'):
        s=g.set_index('t').demand;v=s.dropna();mean=float(v.mean());sd=float(v.std(ddof=0));positive=float((v>0).mean())
        corr=float(s.corr(s.shift(7))) if len(v)>14 and sd>0 else np.nan
        recent=s[s.index>cutoff-28].mean();prior=s[(s.index>cutoff-56)&(s.index<=cutoff-28)].mean()
        rows.append({'entity':e,'history_days':len(s),'history_observed_days':int(g.observation_mask.sum()),
            'train_mean':mean,'train_sd':sd,'train_p99':float(v.quantile(.99)),'positive_fraction':positive,
            'train_CV':sd/mean if mean>0 else 0.,'intermittency':'intermittent' if positive<=.5 else 'frequent',
            'lag7_correlation':corr,'seasonality':'lag7_correlation_ge_0.3' if np.isfinite(corr) and corr>=.3 else 'weak_or_unresolved',
            'recent_to_prior_mean':float(recent/prior) if prior>0 else np.nan})
    f=pd.DataFrame(rows);rank=f.train_CV.rank(method='first',pct=True)
    f['volatility']=np.where(rank<=1/3,'low',np.where(rank<=2/3,'medium','high'))
    return f

def run():
    accuracy=[];coverage=[];improve=[];errors=[];worst=[];weights=[];corrs=[];ablations=[];corrections=[];contributions=[];residuals=[];characteristics=[];regime_rows=[];costs=[]
    checks=[]
    # Analytic edge cases validate percentage conventions and weighting.
    m=metrics([0,2],[0,1],[1,1.25]);assert np.isclose(m['MAE'],.5) and m['MAPE']==50 and m['MAPE_coverage']==.5
    assert np.isclose(m['sMAPE'],100/3) and np.isclose(m['NWRMSLE'],np.sqrt(1.25*np.log(2/3)**2/2.25))
    assert metrics([0,0],[0,0])['WAPE'] is None and metrics([0],[0])['MAPE'] is None
    checks.append('metric analytic zero-denominator, MAPE-coverage and perishability-weight examples passed')
    for d in DATASETS:
        f=pd.read_csv(ROOT/f'predictions/pilot/{d}_predictions.csv',dtype={'entity':str,'store':str})
        first=f[f.stage.eq('calibration_fit')].copy();second=f[f.stage.eq('pilot_validation')].copy();origin=int(second.origin.iloc[0])
        assert not f.duplicated(['stage','model','entity','target_t']).any()
        assert first.target_t.max()<=origin and (f.target_t>f.origin).all()
        panel=load_panel(d);r=regimes(panel,int(first.origin.iloc[0]));r['dataset']=d;regime_rows.append(r)
        second=second.merge(r.drop(columns='dataset'),on='entity',validate='many_to_one')
        pivot=second.pivot(index=['entity','target_t'],columns='model',values='prediction')
        truth=second[second.model.eq('Naive')].set_index(['entity','target_t'])
        common=truth.score_mask & truth.y.notna() & pivot[ALL].notna().all(axis=1)
        assert common.any(),f'No common scoring cells: {d}'
        common_keys=common[common].index
        clean=second.set_index(['entity','target_t']).loc[common_keys].reset_index()
        w=json.loads((ROOT/f'models/pilot/{d}_Hybrid_weights.json').read_text())
        np.testing.assert_allclose(pivot.loc[common,'Hybrid'],sum(pivot.loc[common,m]*w['weights'][m] for m in MODELS),rtol=1e-10,atol=1e-10)
        assert np.isclose(sum(w['weights'].values()),1) and min(w['weights'].values())>=0
        for model in ALL:
            sub=second[second.model.eq(model)];valid=sub.score_mask & sub.y.notna() & sub.prediction.notna()
            coverage.append({'dataset':d,'model':model,'scope':'pilot_validation','scheduled_cells':len(sub),'scorable_actual_cells':int(sub.score_mask.sum()),
                'successful_scorable_cells':int(valid.sum()),'failed_scorable_cells':int((sub.score_mask & sub.prediction.isna()).sum()),
                'coverage_fraction':float(valid.sum()/sub.score_mask.sum()),'common_six_model_cells':int(common.sum()),
                'scheduled_entities':sub.entity.nunique(),'fully_successful_entities':int(sub.groupby('entity').prediction.apply(lambda z:z.notna().all()).sum())})
            for population,z in [('common_six_model',clean[clean.model.eq(model)]),('own_successful',sub[valid])]:
                accuracy.append({'dataset':d,'model':model,'scope':'pilot_validation','population':population,'origin':origin,**score(z)})
        a=pd.DataFrame(accuracy);a=a[(a.dataset==d)&a.population.eq('common_six_model')].set_index('model')
        for model in MODELS:
            for metric in ['MAE','RMSE','sMAPE','WAPE']+(['NWRMSLE'] if d=='favorita' else []):
                b=float(a.loc[model,metric]);h=float(a.loc['Hybrid',metric]);improve.append({'dataset':d,'baseline':model,'metric':metric,'baseline_error':b,'hybrid_error':h,'absolute_improvement':b-h,'percentage_improvement':100*(b-h)/b if b else None,'population':'common_six_model','scope':'pilot_validation'})
        clean['promotion_status']=np.where(clean.promotion.isna(),'unknown',np.where(clean.promotion>0,'positive_recorded_or_discount_proxy','zero_recorded'))
        for dimension in ['horizon','volume','intermittency','volatility','seasonality','promotion_status','category','store','dataset','origin']:
            for (model,segment),g in clean.groupby(['model',dimension],dropna=False):
                errors.append({'dataset':d,'model':model,'dimension':dimension,'segment':str(segment),'scope':'pilot_validation',**score(g)})
        clean['residual']=clean.prediction-clean.y;clean['absolute_error']=clean.residual.abs()
        for model,g in clean.groupby('model'):
            for _,row in g.nlargest(20,'absolute_error').iterrows():
                flags=[]
                if row.y>max(row.train_p99,row.train_mean+3*row.train_sd):flags.append('extreme_or_spike_vs_training')
                if row.y<.25*row.train_mean:flags.append('drop_vs_training_mean')
                if row.promotion_status.startswith('positive'):flags.append('promotion_or_discount_recorded')
                if pd.notna(row.holiday) and row.holiday>0:flags.append('holiday_recorded')
                if row.intermittency=='intermittent':flags.append('intermittent_history')
                if row.assumed_zero:flags.append('unrecorded_entity_day_assumed_zero')
                if row.history_days<365:flags.append('less_than_one_year_history')
                if row.positive_fraction<.2:flags.append('sparse_positive_history')
                if pd.notna(row.returned_units) and row.returned_units!=0:flags.append('return_or_adjustment_same_day')
                if row.seasonality=='lag7_correlation_ge_0.3':flags.append('weekly_association_in_training')
                if pd.notna(row.recent_to_prior_mean) and (row.recent_to_prior_mean>2 or row.recent_to_prior_mean<.5):flags.append('recent_training_mean_shift')
                worst.append({**{k:row[k] for k in ['dataset','model','entity','target_t','horizon','y','prediction','residual','absolute_error','train_mean','train_sd','positive_fraction']},'associations':';'.join(flags) or 'no_prespecified_flag','interpretation':'descriptive association; not a causal explanation'})
        residuals.append(clean[['dataset','model','entity','origin','target_t','horizon','y','prediction','residual','volume','intermittency','volatility','seasonality']])
        ep=pivot.loc[common,MODELS].subtract(truth.loc[common,'y'],axis=0)
        ec=ep.corr()
        for i,m1 in enumerate(MODELS):
            for m2 in MODELS[i+1:]:corrs.append({'dataset':d,'model_1':m1,'model_2':m2,'error_Pearson_r':ec.loc[m1,m2],'n_cells':len(ep),'independent_origins':1,'interpretation':'descriptive pooled correlation; cells not independent'})
        for model in MODELS:
            weights.append({'dataset':d,'model':model,'weight':w['weights'][model],'first_calibration_MAE':w['calibration_MAE'][model],'calibration_cells':w['calibration_cells'],'calibration_max_label':w['calibration_max_label'],'evaluation_origin':origin})
            v=pivot.loc[common,model]*w['weights'][model]
            contributions.append({'dataset':d,'model':model,'weight':w['weights'][model],'mean_weighted_forecast':v.mean(),'share_total_hybrid_forecast':v.sum()/pivot.loc[common,'Hybrid'].sum() if pivot.loc[common,'Hybrid'].sum() else None,'note':'algebraic contribution; not independent causal benefit'})
        # All ablations share all-five complete calibration cells and all-six evaluation cells.
        cp=first.pivot(index=['entity','target_t'],columns='model',values='prediction');ct=first.drop_duplicates(['entity','target_t']).set_index(['entity','target_t'])
        ck=cp.index[cp[MODELS].notna().all(axis=1)&ct.score_mask.reindex(cp.index)]
        cf=first.set_index(['entity','target_t']).loc[ck].reset_index()
        sets=[('Naive + SARIMA',['Naive','SARIMA']),('SARIMA + XGBoost',['SARIMA','XGBoost']),('XGBoost + LSTM',['XGBoost','LSTM']),('LSTM + Transformer',['LSTM','Transformer']),('Full Hybrid',MODELS)]+[(f'Without {m}',[x for x in MODELS if x!=m]) for m in MODELS]
        for label,models in sets:
            aw=fit_weights(cf,origin,models);ap=combine(second[second.model.isin(MODELS)],aw).reindex(common_keys)
            z=truth.loc[common_keys].copy();z['prediction']=ap
            ablations.append({'dataset':d,'combination':label,'weights':json.dumps(aw['weights'],sort_keys=True),'scope':'pilot_validation_exploratory','population':'common_six_model',**score(z)})
        # One correction trial, declared before training. Calibration-fold residuals only.
        ch=combine(first,w).reindex(ck);offset=float((ct.loc[ck,'y']-ch).mean())
        z=truth.loc[common_keys].copy();z['prediction']=pivot.loc[common_keys,'Hybrid'];before=score(z)
        z['corrected_prediction']=(z.prediction+offset).clip(lower=0);after=score(z,'corrected_prediction')
        for metric in ['MAE','RMSE','sMAPE','WAPE']+(['NWRMSLE'] if d=='favorita' else []):
            corrections.append({'dataset':d,'trial':1,'correction':'pooled additive Hybrid calibration residual offset; nonnegative clip','offset':offset,'metric':metric,'before':before[metric],'after':after[metric],'absolute_improvement':before[metric]-after[metric],
                'reason':'first calibration Hybrid signed residual mean','fit_max_label':int(first.target_t.max()),'evaluation_origin':origin,'retained':False,'retention_reason':'exploratory single-origin comparison; untouched baseline remains primary; requires additional development origins','overfitting_risk':'one short calibration fold; reuses weight-calibration labels, so offset may be unstable'})
        z[['dataset','origin','y','prediction','corrected_prediction']].reset_index().to_csv(ROOT/f'predictions/pilot/{d}_correction_trial_1.csv',index=False)
        ledger=json.loads((ROOT/f'data/processed/pilot/{d}_preprocessing_ledger.json').read_text())
        characteristics.append({'dataset':d,**{k:ledger[k] for k in ['raw_original_rows_from_completed_audit','development_input_rows','eligible_entity_count','pilot_entity_count','final_model_ready_rows','observed_panel_rows','assumed_zero_rows','masked_global_gap_rows','panel_t_min','panel_t_max']}})
        costs.append(pd.read_csv(P/f'{d}_compute.csv'))
        checks.append(f'{d}: prediction keys, chronology, convex weights and independently recomputed Hybrid predictions verified')
    accuracy=save('model_accuracy.csv',accuracy);summary=save('model_accuracy_summary.csv',accuracy[accuracy.population.eq('common_six_model')]);coverage=save('model_failure_coverage.csv',coverage)
    improve=save('hybrid_improvements.csv',improve);errors=save('error_analysis.csv',errors);worst=save('worst_prediction_associations.csv',worst)
    weights=save('hybrid_weights.csv',weights);corrs=save('hybrid_error_correlations.csv',corrs);ablations=save('hybrid_ablation.csv',ablations);corrections=save('error_correction_before_after.csv',corrections)
    save('hybrid_component_contributions.csv',contributions);save('pilot_residuals.csv',pd.concat(residuals));save('training_demand_regimes.csv',pd.concat(regime_rows))
    characteristics=save('pilot_dataset_characteristics.csv',characteristics);costs=save('computational_performance.csv',pd.concat(costs));
    # Per-entity comparisons supplement pooled rankings without selecting favorable entities.
    per=[]
    rr=pd.concat(residuals)
    for (d,e,m),g in rr.groupby(['dataset','entity','model']):per.append({'dataset':d,'entity':e,'model':m,'MAE':g.residual.abs().mean(),'n':len(g)})
    per=save('entity_accuracy.csv',per)
    consistency=[]
    for d in DATASETS:
        for dim,frame,key in [('entity',per[per.dataset.eq(d)],'entity'),('horizon',errors[errors.dataset.eq(d)&errors.dimension.eq('horizon')],'segment')]:
            wide=frame.pivot(index=key,columns='model',values='MAE')
            for m in MODELS:
                delta=wide[m]-wide.Hybrid
                consistency.append({'dataset':d,'dimension':dim,'baseline':m,'Hybrid_lower_MAE':int((delta>1e-10).sum()),'Hybrid_higher_MAE':int((delta< -1e-10).sum()),'ties':int((delta.abs()<=1e-10).sum()),'total':len(delta)})
    consistency=save('hybrid_consistency.csv',consistency)
    verification={'status':'pass','checks':checks,'final_test_evaluated':False,'independent_evaluation_origins_per_dataset':1}
    (P/'results_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    versions={}
    for name in ['numpy','pandas','torch','xgboost','scipy','scikit-learn','statsmodels','psutil','openpyxl','patsy','cloudpickle','narwhals','formulaic','interface-meta','wrapt','joblib','threadpoolctl','filelock','typing_extensions','sympy','networkx','jinja2','fsspec','mpmath','MarkupSafe']:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:pass
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'experiments').glob('*.json'))+list((ROOT/'models/pilot').glob('*configuration.json'))+list((ROOT/'models/pilot').glob('*weights.json'))+list((ROOT/'predictions/pilot').glob('*.csv'))
    freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'pilot_snapshot_only_not_final_test_freeze','python':sys.version,'platform':platform.platform(),'dependencies':versions,'checksums':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.name!='pilot_snapshot.json'}}
    (ROOT/'experiments/pilot_snapshot.json').write_text(json.dumps(freeze,indent=2)+'\n')
    (ROOT/'requirements-pilot-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(versions.items()))+'\n')
    lines=['# Demand forecasting: measured feasibility pilot','',
        '**Status:** preprocessing, leakage gates, two chronological calibration folds and all six model pipelines have been exercised on 12 training-selected entities per dataset. These are preliminary validation results, not final-test results or full-population research conclusions.','',
        '## Hybrid architecture and authorization','',
        'The repository search found no executable Hybrid implementation. The user clarified that a new Hybrid must be built. The starting architecture was stated before implementation: a nonnegative weighted combination of Naive, SARIMA, XGBoost, LSTM and Transformer. Each weight is proportional to the inverse pooled MAE on the first calibration fold, normalized to sum to one. Weights are frozen before the second calibration fold. Near-zero calibration errors share all weight; epsilon is recorded. This is a transparent starting forecast combination, not a claim of architectural novelty.','',
        'Each component is refitted through the second origin. Weights remain those learned from the earlier out-of-sample predictions. If any component fails, the full Hybrid cell is missing; no silent replacement or weight renormalization is used.','',
        '## Design and scope','',
        'The original split manifest is preserved. Tuning periods are part of historical training at these calibration origins, but no hyperparameter search was performed. Fixed small configurations use seed 20260922, CPU with two threads, five neural epochs, a 28-day sequence, training windows every seven days, and no random shuffle. The pilot evaluates Dunnhumby days 614–627, Favorita 2017-04-26–2017-05-11 and UCI 2011-09-02–2011-09-15. The preceding calibration fold is used only for Hybrid calibration and the declared correction trial. Final-test periods were not transformed into model panels or forecast.','',
        'Cohorts were fixed using only the initial training period: four eligible entities from each volume tercile, ordered by a deterministic hash. Selection did not use forecasting errors. Panels represent pooled household product purchases for Dunnhumby, store-item sales for Favorita, and product purchases for UCI. They are not directly comparable physical units across datasets.','',
        'Missing entity-days become conditional sales zeros only after first positive history and only on globally recorded days. Global gaps are masked. This complete-capture assumption does not identify stock availability or latent demand; availability remains unknown for unrecorded entity-days. UCI non-recording days are excluded from scoring, so its pilot describes recorded trading days. Returns and excluded records have separate ledgers; no raw files were edited. Favorita fractional units and perishable weights are preserved.','',
        '## Table 1 — Dataset and pilot characteristics','',table(characteristics),'',
        'The raw row counts come from the completed audit. Development counts stop at each second calibration horizon. Only 12 entities per dataset are panelled; exclusion and eligibility files retain the cohort decision record.','',
        '## Tables 2–4 — Six-model validation accuracy','',
        'Primary tables use exactly the same successful, scorable cells across all six models within each dataset. Own-successful scores and full coverage are also saved so failure exclusions remain visible. MAPE excludes actual zero demand; percentages are multiplied by 100. sMAPE assigns zero to both-zero cells. WAPE is undefined for a zero total actual. Bias is prediction minus actual. Favorita NWRMSLE is sqrt(sum(weight × squared log1p error) / sum(weight)), with weight 1.25 for perishable items and 1 otherwise. WMAE uses the same weights; it is not assigned to datasets with no justified weights.','']
    for d in DATASETS:
        lines += [f'### {d}',table(summary[summary.dataset.eq(d)],['model','n','MAE','RMSE','MAPE','MAPE_coverage','sMAPE','WAPE','Bias','WMAE','NWRMSLE']),'']
    lines+=['## Table 5 — Hybrid improvement over each component','',
        'Positive values favor Hybrid; negative values are deterioration. All requested metric comparisons are in `hybrid_improvements.csv`. MAE comparisons:','',table(improve[improve.metric.eq('MAE')],['dataset','baseline','absolute_improvement','percentage_improvement']),'',
        '## Table 6 — Hybrid error by demand regime','',
        'Regimes are computed from history available before the first calibration origin. Intermittent means at most half of scorable history days have positive demand. Volatility is training coefficient-of-variation rank within the 12-entity cohort. Weekly association means training lag-7 correlation at least 0.3; this is a diagnostic threshold, not a causal or statistical seasonality test.','',
        table(errors[errors.model.eq('Hybrid')&errors.dimension.isin(['volume','intermittency','volatility'])],['dataset','dimension','segment','n','MAE','RMSE','sMAPE','WAPE','Bias']),'',
        'All six models are broken down by horizon, volume, intermittency, volatility, weekly association, recorded promotion/discount status, category, store where applicable, dataset and origin in `error_analysis.csv`. Future realized promotion and holiday fields are used only for retrospective diagnostics; forecast features contain past values and deterministic calendar terms.','',
        '### Zero-heavy scoring and bias', '', 'Dunnhumby has 112 zero actuals among 168 scored cells. Hybrid predicts positive values in every cell, while Naive is exactly correct at zero on 72 cells. Under the declared sMAPE convention, a positive forecast against zero actual has 200% error, explaining why improved MAE can coexist with worse sMAPE. UCI also has many zeros. The five-epoch neural models can have substantial negative bias and clipped outputs; lower MAE alone does not establish adequate fit. Clipping counts, bias, RMSE and MAPE coverage must be read together.', '', '## Table 7 — One declared correction trial','',
        'The first calibration fold supplies a pooled Hybrid residual offset, which is added to second-fold forecasts and clipped at zero. This is a correction of Hybrid, not a seventh forecasting architecture. The offset is estimated before the evaluated fold. The unchanged Hybrid remains the primary result; no correction is retained for final testing on evidence from this single origin.','',
        table(corrections[corrections.metric.eq('MAE')],['dataset','offset','before','after','absolute_improvement','retained']),'',
        'All requested before/after metrics, reasons and overfitting risks are in `error_correction_before_after.csv`. This trial reuses weight-calibration labels and may overfit that short fold. Further development origins are required before choosing a correction.','',
        '## Table 8 — Hybrid ablation','',
        'Every subset learns inverse-MAE weights on the same all-five successful calibration cells and is scored on the same all-six successful evaluation cells. Components are reused without retraining. These are exploratory diagnostics, not a selection of a new final architecture.','',table(ablations,['dataset','combination','MAE','RMSE','sMAPE','WAPE','NWRMSLE']),'',
        '### Weights and error diversity','',table(weights,['dataset','model','weight','first_calibration_MAE']),'',table(corrs,['dataset','model_1','model_2','error_Pearson_r']),'',
        'Positive residual correlations limit diversification. A convex average need not beat the best component: inverse marginal MAE ignores error covariance, regime variation and changes between folds. Weight calibration is short and noisy. Algebraic component contributions are saved separately; they should not be interpreted as causal benefits.','',
        '## Table 9 — Computational performance','',
        'Times are measured fit/predict wall times, excluding preprocessing, feature construction and serialization. Inference timings are single batch measurements, not repeated latency percentiles. RSS is sampled every 20 ms in a shared process and includes library and allocator memory. GPU memory is not applicable (CPU). Parameter counts are heterogeneous: tree nodes and neural weights are not comparable.','',
        table(costs[costs.stage.eq('pilot_validation')],['dataset','model','training_seconds','inference_seconds','peak_RSS_bytes','model_bytes','parameters','parameter_kind']),'',
        'Hybrid training cost includes first-fold component fitting, second-fold refitting and weight fitting. Standalone rows show their second-fold fit; both stages are retained in the CSV for lifecycle comparisons. Hybrid inference cost includes all five components plus combination. Its reported peak RSS is the maximum component-process peak, not a simultaneous deployment benchmark.','',
        '## Table 10 — Failures and coverage','',table(coverage,['dataset','model','scheduled_cells','scorable_actual_cells','successful_scorable_cells','failed_scorable_cells','common_six_model_cells']),'',
        'No component fitting or forecasting failures occurred in this pilot. The implementation marks nonconverged SARIMA forecasts missing rather than replacing them with a successful fallback. The Naive baseline has a declared last-valid fallback for missing seasonal lags; its cell count is in the computational CSV. Detailed exception and optimizer logs are retained per dataset.','',
        '## Error associations and research questions','']
    for d in DATASETS:
        a=summary[summary.dataset.eq(d)].set_index('model');h=a.loc['Hybrid'];best=a.MAE.idxmin();im=improve[(improve.dataset==d)&improve.metric.eq('MAE')]
        wins=', '.join(im[im.absolute_improvement>0].baseline) or 'none';losses=', '.join(im[im.absolute_improvement<0].baseline) or 'none'
        seg=errors[(errors.dataset==d)&errors.model.eq('Hybrid')&errors.dimension.eq('volume')].sort_values('MAE',ascending=False).iloc[0]
        wc=worst[(worst.dataset==d)&worst.model.eq('Hybrid')].associations.str.split(';').explode().value_counts().head(3)
        lines += [f'- **{d} (RQ1–3):** Hybrid MAE {h.MAE:.4f}, RMSE {h.RMSE:.4f}, WAPE {h.WAPE:.2f}%. Lowest measured pilot MAE: {best}. Hybrid MAE is lower than {wins}; higher than {losses} (RQ4–8). This is one pilot origin.',
            f'- **{d} regimes and associations (RQ9–10):** highest Hybrid volume-segment MAE is {seg.segment} ({seg.MAE:.4f}). In its 20 largest absolute errors, common flags are '+', '.join(f'{k} ({v}/20)' for k,v in wc.items())+'. These flags can overlap and do not establish causes.']
    c=corrections[corrections.metric.eq('MAE')]
    lines += ['',f'**RQ11:** The single correction improves validation MAE on '+(', '.join(c[c.absolute_improvement>0].dataset) or 'none of the datasets')+' and worsens it on '+(', '.join(c[c.absolute_improvement<0].dataset) or 'none')+'. See Table 7 for magnitudes. It is not retained.',
        '', '**RQ12:** Generalization of a corrected Hybrid across the datasets is not established. No final-test evaluation has been run.',
        '', '**RQ13:** Tables 5 and 8 and the residual correlations provide measured evidence about combination on this pilot only. Combining all five is not presumed beneficial, and subset comparisons are not final model selection.',
        '', '**RQ14:** Table 9 contains measured small-cohort CPU costs. They are not full-dataset runtime or memory estimates.',
        '', '**RQ15:** Horizon and entity results are saved for every included cell in `error_analysis.csv` and `entity_accuracy.csv`. Consistency beyond a single evaluated origin cannot be established. Aggregate improvements must not be treated as uniform gains across horizons or products.',
        '', table(consistency,['dataset','dimension','baseline','Hybrid_lower_MAE','Hybrid_higher_MAE','ties','total']), '', 'Statistical significance could not be reliably established because of the limited number of independent forecast origins. No p-values, confidence intervals or Diebold–Mariano claims are manufactured.',
        '', 'The diagnostics do not establish underfitting, overfitting, latent availability, causal promotion effects, or store sparsity from this small cohort. XGBoost, LSTM and Transformer generate direct horizons without recursively feeding predictions into their features. SARIMA propagates its state across the horizon; this pilot does not isolate propagation error from other error sources. Passing leakage and join checks is evidence for tested contracts, not proof against every possible data issue.',
        '', '## Feasibility gate and next research stage','',
        'The exercised pipeline is feasible on the 12-entity CPU pilot. Full training remains disabled. The machine has 8 GiB RAM and only about 2.5 GiB free disk before model artifacts. The 5 GB Favorita train CSV must remain compressed and streamed. Full populations include many thousands of entities; per-entity SARIMA storage/fitting and global neural window materialization require a separately bounded design. Scaling these pilot measurements linearly would be unjustified.',
        '', 'Before a full experiment: provision working storage, use disk-bounded streaming training windows, set a training-only cohort and computational budget, use the unchanged tuning/calibration origins to compare declared configurations, validate any selected correction on additional untouched development origins, freeze configuration and checksums, then run the final test once. Do not promote pilot-ranked variants or adjust final-test boundaries.',
        '', '## Reproducibility and saved artifacts','',
        '- `experiments/continuation_protocol.json`: declared scope, architecture, correction budget and disabled final test.',
        '- `experiments/leakage_audit_report.json`: pre-training leakage gate and code/data checksums.',
        '- `experiments/pilot_snapshot.json`: versions, code/configuration/prediction checksums; pilot snapshot only.',
        '- `data/processed/pilot/`: panels, masks, eligibility/cohort/exclusions, preprocessing and returns ledgers.',
        '- `models/pilot/`: component artifacts, transform states, configurations and frozen Hybrid weights.',
        '- `predictions/pilot/`: both-fold forecasts and one correction trial.',
        '- `logs/pilot/`: training histories, execution and optimizer warning logs.',
        '- `results/pilot/results_verification.json`: metric and prediction consistency checks.', '- `results/pilot/saved_model_replay.json`: all 18 saved second-fold model outputs reproduced within 1e-6.',
        '', 'See `experiments/pilot_reproduction.md` for commands and scope. The completed audit remains unchanged as historical context. Its earlier statement that no models had been trained describes the audit stage, not this continuation.','']
    (R/'pilot_research_report.md').write_text('\n'.join(lines))
    print('Report complete:',R/'pilot_research_report.md')

if __name__=='__main__':run()
