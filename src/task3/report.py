from setup import *

def table(d):return d.to_markdown(index=False,floatfmt='.5g')
def main():
    r=pd.read_csv(O/'model_results_by_year.csv');summary=pd.read_csv(O/'model_summary.csv');gain=pd.read_csv(O/'incremental_gain_summary.csv');strict=pd.read_csv(O/'incremental_gain_summary_strict_3yr.csv');effort=pd.read_csv(O/'incremental_gain_summary_primary_1yr.csv');paired=pd.read_csv(O/'paired_feature_gain_by_year.csv');candidates=pd.read_csv(O/'candidate_audit.csv');audit=json.loads((O/'effort_proxy_audit.json').read_text());miss=pd.read_csv(O/'missing_features_primary_3yr.csv');co=pd.read_csv(O/'logistic_coefficients_primary_3yr.csv');cal=pd.read_csv(O/'calibration_bins.csv')
    cs=co[co.feature_group=='M2'].groupby('feature').agg(mean_beta_per_train_SD=('beta_per_train_SD','mean'),min_beta=('beta_per_train_SD','min'),max_beta=('beta_per_train_SD','max'),years_positive=('beta_per_train_SD',lambda s:int((s>0).sum())),curvature_low_min=('curvature_CI_low','min'),curvature_high_max=('curvature_CI_high','max')).reset_index();cs.to_csv(O/'logistic_coefficient_summary.csv',index=False)
    def row(df,model,comp):return df[(df.model==model)&(df.comparison==comp)].iloc[0]
    def supported(comp):
        return [model for model in ['Logistic','RF'] if row(gain,model,comp).mean_delta_PR_AUC>0 and row(gain,model,comp).median_delta_PR_AUC>0 and row(gain,model,comp).years_positive>=5 and row(strict,model,comp).mean_delta_PR_AUC>0 and row(effort,model,comp).mean_delta_PR_AUC>0]
    extreme=supported('M2-M1');ordinary=supported('M1-M0');m0=summary[summary.feature_group=='M0'];decision=('A — PROCEED: CLIMATE EXTREMES' if extreme and ordinary else 'B — PROCEED: GENERAL CLIMATE' if ordinary else 'C — MODIFY: SPATIAL / LANDSCAPE' if m0.median_AP_over_prevalence.max()>2 else 'D — ABANDON SALSOLA FORECASTING')
    review=json.loads((O/'decision_review.json').read_text()) if (O/'decision_review.json').exists() else {}
    if review:decision=review['recommendation']
    # Decision text is reviewed against per-year results, not treated as a significance test.
    (O/'decision_summary.json').write_text(json.dumps(dict(recommendation=decision,extreme_support_models=extreme,ordinary_support_models=ordinary,rule='Final computational assessment integrates all three analyses; not independent peer review'),indent=2))
    n=r.drop_duplicates('prediction_year');cq=candidates[(candidates.quality=='primary')&(candidates.year>=2018)]
    decision_table=[]
    for q,comp in [('M1 > M0?','M1-M0'),('M2 > M1?','M2-M1')]:
        vals=[f'{a.model}: mean ΔAP={a.mean_delta_PR_AUC:.5f}, {a.years_positive}/8 positive, CI[{a.CI_low:.5f},{a.CI_high:.5f}]' for a in gain[gain.comparison==comp].itertuples()];decision_table.append([q,'; '.join(vals)])
    decision_table.extend([['Spatial M0 beats prevalence?',f'Yes: median AP/prevalence Logistic={m0[m0.model=="Logistic"].median_AP_over_prevalence.iloc[0]:.2f}; RF={m0[m0.model=="RF"].median_AP_over_prevalence.iloc[0]:.2f}. M0 includes effort.'],['M2 > M1 in a majority of years?','Primary: 6/8 for both models. Strict: Logistic7/8, RF3/8.'],['RF and Logistic directionally agree?','Primary yes; strict-coordinate extreme increment no.'],['Effort control changes result?','General climate persists: RF7/8 to8/8 positive years; extreme RF interval crosses zero with previous-year effort.'],['Strict coordinates support the same conclusion?','General climate yes; extreme-climate conclusion only for Logistic, RF reverses.'],['Worth continuing?','B: pursue general-climate forecasting cautiously; do not force an extremes headline.']])
    radius_path=O/'radius_incremental_gains.csv'
    radius_text=table(pd.read_csv(radius_path)[['radius_km','model','comparison','mean_delta_PR_AUC','median_delta_PR_AUC','CI_low','CI_high','years_positive']]) if radius_path.exists() else 'Conditional radius check not yet run.'
    coeffnote='Intervals are penalized likelihood curvature intervals on standardized sampled-training coefficients. They ignore repeated-cell/spatial/year dependence and model-selection uncertainty; they are descriptive, not robust confidence intervals or causal tests. Every rolling-year coefficient and interval is retained in logistic_coefficients_primary_3yr.csv.'
    text=f'''# Task 3-Pivot — Climate and Spatial Expansion Forecasting

Minimal pilot, fixed 4-km California grid and 50-km candidates. **Recommendation: {decision}.** The conclusion below must be read with paired-year uncertainty and sensitivity results; all yearly counterexamples are retained. PR-AUC means **average precision (AP)** throughout.

## 1. Objective

Determine whether preceding-year ordinary climate (M1) improves forecasts beyond spatial history and observation effort (M0), and whether five pre-specified extreme summaries (M2) add further information. Predict **newly observed cells**, not biological arrival. No causal claims.

## 2. Reused Data From Tasks 1–2

Reused the 26,294-cell California 4-km grid, cleaned Salsola tragus occurrence records, independent coordinate-quality flags, first-observation logic, candidate labels, GBIF effort audit references, and 9 raw 2016–2024 gridMET wind-speed files. No Task 1/2 analysis rerun. Prior-file SHA256 inventory is logs/task3/prior_tasks_manifest.json; final validation checks preservation.

## 3. Prediction Target

For each year t, exclude any cell observed before t. Positive if first observed during t, negative if still unobserved in t even if observed later. Candidate center must be<=50 km from a prior occupied cell. All event-year histories include pre-2012 occurrences. Main evaluation has **{int(n.n_test.sum()):,} cell-years and{int(n.n_positive_test.sum())} positives**, with prevalence{n.positive_rate.min():.3%}–{n.positive_rate.max():.3%}. Full unsampled test sets:

{table(cq[['year','n_candidates','n_positive','n_sources']])}

## 4. Temporal Leakage Controls

Each historical training row is reconstructed using its own pre-year source history, preceding calendar-year climate and past-only effort. Climatology ends2010. Training labels begin2012; evaluation2018–2025. Feature date assertions cover every row. Imputation and scaling are fitted on sampled past training rows only. No2026 target year, same-year effort or realized test-year weather. No outcome-selected Task2 matching pool used as a test domain. Event-time reconstruction does **not** guarantee historical database availability: current GBIF snapshots contain retrospective additions and corrections.

## 5. Climate Data

Official gridMET NCSS California bounding box:32.4–42.1N,124.5–114.1W, native horizontal stride1. Four variables only: Tmax, precipitation, VPD and nondirectional wind speed. Each annual response is validated against every expected day before caching.176 variable-year inputs from 1981–2024; 9 wind files reused. Sources and checksums: logs/task3/climate_manifest_*.json. Native-grid center nearest to each California grid center, not microclimate or agricultural weighting. Missing coastal/native cells are retained in test evaluation and imputed using training medians.

Candidate-level missing fractions by year/feature are in outputs/task3/missing_features_primary_3yr.csv. Across feature/year entries the maximum missing fraction is **{miss.drop(columns=['prediction_year']).max().max():.3%}**. This missingness is spatially structured and may reduce coastal reliability.

## 6. Climate Feature Construction

M1: preceding-year mean Tmax(C), total precipitation(mm), mean VPD(kPa), mean wind speed(m/s). M2: counts above fixed cell-specific1981–2010 pooled daily p95 for Tmax,VPD,wind; longest within-year precipitation<1mm dry run; precipitation total minus1981–2010 mean annual total. Quantiles use linear interpolation; no moving/seasonal threshold tuning. Missing daily values make the corresponding annual feature missing rather than zero. Baseline daily mean,SD,p95 and annual-total mean/SD are preserved. No Tmin,seasonal variants,landscape features,or additional classifiers. Definitions: outputs/task3/feature_dictionary.csv. Pre2018 correlations are saved; correlated predefined variables were retained and regularization used, not selected by test results.

## 7. Spatial Features

M0: nearest prior source distance; distinct source-cell counts25/50km; sum exp(-distance/25km) over all historical occupied cells; log past effort. Sources are unique occupied cells, not raw occurrence counts. Coordinates EPSG3310,center-to-center distances. Each training/test year has distinct features; never reuse a2025 source map backward.

After the minimal experiment, the user section33 condition was met by ordinary-climate added value. Ran only the prelisted25/100km radii with all original models/features/settings;50km remains primary. The pre-check clarification is outputs/task3/conditional_radius_protocol.md. Results:

{radius_text}

RF M1-M0 is positive in7/8,7/8,6/8years at25,50,100km;100km uncertainty overlaps zero. RF M2-M1 is negative on average at25km and approximately zero at100km. Logistic M2 gains at25/100km also have intervals crossing zero. This strengthens the distinction between general-climate value and uncertain extreme-feature value; no radius was selected as a replacement primary. Training negatives are re-sampled at the same fixed inclusion probability within each domain, so radius differences include finite training-sample variation.

## 8. Observation-Effort Control

Task 2 exact effort existed only for 1,479 outcome-selected matching cells and could not define an honest full-domain test set. Built a uniform 420,704-cell-year vascular-plant effort panel for 2009–2024 across all 26,294 grid cells. GBIF density vector tiles atzoom8,verbose year counts,taxon7707728,countryUS; each year's count uses only that year's property, not total counts across future years. Remove duplicated tile buffers via tile ownership.45 tiles cover every grid corner. Quantization is305.75 Web-Mercator meters (approximately227–258m on the ground acrossCalifornia); this is an approximate effort proxy, not exact survey effort.

Initial adhoc zoom6 tiles were rejected before modeling because coarse geohash aggregation failed audit (rho≈0.004). Final density counts independently compared with exact Task2 records for2015–2024: Spearman correlation **{min(x['spearman_log_count'] for x in audit['yearly']):.3f}–{max(x['spearman_log_count'] for x in audit['yearly']):.3f}**, aggregate count ratios **{min(x['total_ratio'] for x in audit['yearly']):.3f}–{max(x['total_ratio'] for x in audit['yearly']):.3f}**. All years passed pre-outcome rho>=.95 and ratio0.9–1.1. Never replace only matched cells with exact counts. Density and search snapshots/filter handling differ slightly; audit measures this jointly with spatial quantization. Counts approximate reporting intensity, not standardized botanical survey exposure.

## 9. Rolling Temporal Validation

For every test year 2018–2025, refit using all labels from 2012 through t−1. Fixed hyperparameters, no validation search: Logistic C=1,L2,max_iter2000; RF500trees,max_depth12,min_leaf10,max_features=sqrt. All positives and independent5% of training negatives retained; fixed seed20261004,identical rows across feature groups. No class weighting. Predicted odds multiplied by0.05 to correct known case-control sampling. This correction alone does not establish calibration. Test sets are complete. Precision/recall/F1 use pre-specified0.5 corrected-probability threshold; these can be zero in a rare-event task and were not tuned on test data. Accuracy is not used.

## 10. M0 Spatial Results

{table(m0)}

AP/prevalence measures ranking lift over random expectation, not causal ecological value. M0 includes effort, so its entire gain cannot be attributed to biological spread. No effort-free attribution experiment was added to this minimal run.

## 11. M1 Climate Added Value

{table(gain[gain.comparison=='M1-M0'])}

Interpret relative to M0 of the same algorithm and year, not a different chosen model. All gains use full unsampled held-out candidate sets.

## 12. M2 Climate-Extreme Added Value

{table(gain[gain.comparison=='M2-M1'])}

This is the primary research comparison. The primary results are favorable: both models improve in6/8years with positive paired-mean intervals. However, this is not uniformly replicated: strict-coordinate RF reverses (mean−0.000998,3/8positive), while strict Logistic remains positive (mean+0.001335,7/8positive). Primary RF gains are largest in2018 and2024; omitting any single year leaves a positive mean, so the problem is coordinate/model sensitivity rather than one-year dependence alone. These counterexamples prevent a common robust extremes headline. Pre-specified optional-ablation trigger is in protocol.md; failed triggers are not rescued by new thresholds.

## 13. Logistic Regression Results

{table(summary[summary.model=='Logistic'])}

Standardized M2 coefficient summaries:

{table(cs)}

{coeffnote}

## 14. Random Forest Results

{table(summary[summary.model=='RF'])}

No impurity-importance plot is used as evidence. The pre-outcome no-directional-reversal criterion fails for strict-coordinate RF, so held-out permutation importance, extreme-category ablations and absolute/anomaly extensions were not triggered. The separate user section33 general-forecast condition did trigger25/100km radius checks, reported in section7. Selecting only Logistic to activate a favorable extreme-climate story would conceal the RF counterexample. The main analysis remains the requested minimal50-km pilot; additional radii are sensitivity analyses, not alternate primary selections.

## 15. Per-Year Paired Comparisons

{table(paired)}

Equal-year paired mean and median;20,000 resamples of the8pairedyears give percentile95% intervals for mean gain. Leave-one-year-out ranges in incremental_gain_summary.csv show whether one year drives the mean. Eight correlated forecast origins do not support strong asymptotic inference; bootstrap intervals are descriptive and do not remove shared-training or spatial dependence. Full 48-row primary performance table includes AP,ROC-AUC,precision,recall,F1,Brier,prevalence and lift.

## 16. Coordinate-Quality Sensitivity

Independently reconstruct sources, first-observation labels and candidates from known coordinate uncertainty<=1km. This changes the target sample as well as source quality, so identical scores are not expected. No test positives are transferred from the primary definition.

{table(strict)}

## 17. Observation-Effort Sensitivity

Replace past3-year mean effort with preceding-year effort in every group while keeping labels/domain/model settings fixed. This is still prospective. No same-year effort model is presented as a forecast.

{table(effort)}

## 18. Calibration

{table(r.groupby(['model','feature_group']).agg(mean_Brier=('Brier','mean'),mean_Brier_skill_vs_test_prevalence=('Brier_skill_vs_test_prevalence','mean'),mean_precision_at_0_5=('precision','mean'),mean_recall_at_0_5=('recall','mean')).reset_index())}

Brier reference here is the constant held-out prevalence, a diagnostic reference that is not a deployable trained model. FigureT3_calibration pools held-out predictions into equal-count bins. Low absolute Brier scores are expected at<0.3%prevalence and are not sufficient calibration evidence. The top-risk decile is overpredicted, especially for Logistic M2; RF is closer but still overpredicts. Brier skill against the test-prevalence diagnostic is small for RF (mean about0.3–0.6%) and negative for Logistic (about−2.1% to−6.3%). All precision/recall/F1 at0.5 are zero; this fixed threshold is unsuitable for operational detection at this prevalence and was deliberately not tuned on test labels. Risk maps show ranking scores; do not assume trustworthy absolute probabilities for deployment.

## 19. Interpretation

{table(pd.DataFrame(decision_table,columns=['Question','Numerical evidence']))}

Interpret predictive information only. Annual climate may represent persistent spatial habitat gradients, geographic sampling behavior or year-specific conditions. This pilot does not separate those mechanisms. Local p95 extreme features use historical reference values, but do not by themselves identify unusual-condition effects independently of geography. A negative result does not show climate has no ecological role; it shows insufficient added value in this fixed data/scale/window/model test.

## 20. Limitations

Only 8 forecast years and 318 primary positive cell-years,small early training sets; GBIF observer bias,unknown uncertainty in the primary set,taxonomic issues preserved fromTask1,and current-snapshot backfilling. Negatives are not verified absences. Spatially quantized effort and nearest-gridMET matching; coastal missingness is imputed. No spatial holdout; nearby training/test cells can share habitat and observer patterns. Official gridMET documentation warns about precipitation-source inhomogeneity affecting frequency/intensity and coarse parent wind fields; dry-spell/extreme counts need caution. Archived climate products have also been reprocessed, so this is an event-date hindcast using current data versions, not a reconstruction of every product version available at each forecast origin. [gridMET limitations](https://www.climatologylab.org/gridmet.html). Test-year occurrence labels are necessarily known for evaluation but never enter that year's predictors. Annual previous-year climate can miss shorter biological windows. Collinear features and repeated cells limit coefficient interpretation. Fixed models and5%negative sampling are a feasibility choice,not an optimized benchmark. Calibrated decision support would require further validation.

## 21. Comparison With Task 2 Directional Wind Result

Task2 remains **NO SIGNAL**: primary equal-year alignment difference0.00160,95%CI[−0.0191,0.0223],4/8positiveyears,rotationp≈0.375; adjusted standardizedalignmentcoefficient≈0.00215,p≈0.975. No directional wind model or threshold search was restarted. Task3 tests a different predictor class and forecasting target comparison. Only supported climate gains can justify claiming broad environmental predictors are more useful under these tested scales; Task2's negative finding is retained regardless of Task3.

## 22. Go / No-Go Recommendation

**{decision}** — conditional continuation of the general-climate direction, not a strong extreme-climate recommendation.

The evidence supports **ordinary climate added value**, most clearly in RF. Its mean AP increases from0.015576(M0) to0.022211(M1), a42.6%relative increase in equal-year mean AP; paired mean gain0.006635,95%year-bootstrapCI[0.002067,0.011529],7/8years positive. This ordinary-climate direction also holds under strict coordinates(7/8) and preceding-year effort(8/8). Logistic ordinary-climate improvements are smaller and its intervals cross zero, so the strength is algorithm dependent.

Do **not** interpret B as proving M2 has no value: Logistic extreme increments survive both sensitivities. The primary RF increment is positive too. But strict RF reverses, making extremes a secondary, unsettled finding. This is why A is not recommended. C and D would discard the reproducible general-climate ranking gain.

Suggested working title: **Climate-Informed Spatiotemporal Forecasting of Newly Observed Russian Thistle Cells in California**. A defensible story is: spatial history and reporting effort provide a useful baseline; preceding-year general climate improves RF ranking across most future years; extreme summaries offer inconsistent additional value across coordinate quality/model choices. Preserve the Task2 negative directional-wind experiment. Do not claim climate causes invasion or that predicted scores identify true biological arrivals.

Next proposed confirmation, **not executed automatically**: freeze a general-climate analysis; assess genuine spatial transfer with spatially separated holdouts; distinguish dynamic anomalies from stable habitat/reporting geography; compare observer-effort-only ranking; assess probability calibration; check literature novelty before a full AIDA manuscript. These are conditions for scaling, not proven results.

Decision is a pilot research recommendation,not publication acceptance or causal proof. Preserve all years and sensitivities. Do not expand automatically to landscape data,new species,extra weather thresholds or deep models. If climate gains are weak, a spatial/landscape pivot is a proposed next experiment, not already established improvement.

Reproducibility: README_task3.md,requirements_task3.lock.txt,src/task3,raw response metadata,processed feature panels,all fold predictions,and logs/task3/validation.json. Figures:

- ../figures/T3_model_ablation_prauc.png
- ../figures/T3_climate_gain_by_year.png
- ../figures/T3_extreme_gain_by_year.png
- ../figures/T3_2025_expansion_risk_map.png
- ../figures/T3_calibration.png
- ../figures/T3_robustness_summary.png (additional sensitivity summary)

Data references: [gridMET official description](https://www.climatologylab.org/gridmet.html),[gridMET NCSS catalog](https://tds-proxy.nkn.uidaho.edu/thredds/catalog.html),[GBIF Maps API](https://techdocs.gbif.org/en/openapi/v2/maps),[GBIF occurrence API](https://techdocs.gbif.org/en/openapi/v1/occurrence). Exact URLs/retrieval times/checksums are retained with each cached response. No literature novelty claim was evaluated in this computational pilot.
'''
    (R/'outputs/task3_pivot_feasibility_report.md').write_text(text)
if __name__=='__main__':main()
