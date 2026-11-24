from config import *
def main():
    a=pd.read_csv(O/'decomposition_results_by_year.csv');b=pd.read_csv(O/'spatial_temporal_results_by_year.csv');c=pd.read_csv(O/'climate_decomposition_results_by_year.csv');ag=pd.read_csv(O/'decomposition_gain_summary.csv');bg=pd.read_csv(O/'spatial_temporal_gain_summary.csv');cg=pd.read_csv(O/'climate_decomposition_gain_summary.csv')
    rows=[]
    for model in ['Logistic','RF']:
        for group in ['M0','M1']:
            for buf in [0,25]:
                av=a[(a.model==model)&(a.feature_group==group)];bv=b[(b.model==model)&(b.feature_group==group)&(b.buffer_km==buf)];rows.append(dict(model=model,feature_group=group,buffer_km=buf,temporal_AP=av.PR_AUC.mean(),spatial_temporal_AP=bv.PR_AUC.mean(),climate_gain_temporal=ag[(ag.model==model)&(ag.comparison=='M1-M0')].mean_delta_AP.iloc[0],climate_gain_spatial_temporal=bg[(bg.model==model)&(bg.buffer_km==buf)].mean_delta_AP.iloc[0],summary='equal-year meanAP,not pooled'))
    pd.DataFrame(rows).to_csv(O/'spatial_generalization_table.csv',index=False)
    cc=pd.read_csv(O/'climate_decomposition_summary.csv');cc['years_gain_positive']=[np.nan if r.feature_group=='C0' else int(cg[(cg.model==r.model)&(cg.comparison==dict(CN='CN-C0',CA='CA-C0',CNA='CNA-CN')[r.feature_group])].years_positive.iloc[0]) for r in cc.itertuples()];cc.to_csv(O/'climate_decomposition_table.csv',index=False)
    folds=pd.read_csv(O/'B_fold_results.csv');wide=folds.pivot(index=['prediction_year','spatial_fold','buffer_km','model'],columns='feature_group',values='PR_AUC').reset_index();wide['delta_AP']=wide.M1-wide.M0;wide.to_csv(O/'spatial_fold_gains_by_year.csv',index=False);fs=wide.groupby(['spatial_fold','buffer_km','model']).agg(mean_delta_AP=('delta_AP','mean'),years_AP_defined=('delta_AP','count'),years_positive=('delta_AP',lambda s:int((s>0).sum()))).reset_index();fs.to_csv(O/'spatial_fold_gain_summary.csv',index=False)
    warnings=[]
    for p in (O/'predictions').glob('*.json'):
        d=json.loads(p.read_text())
        if d.get('warnings') or d.get('single_class_fallback'):warnings.append({k:d[k] for k in ['phase','prediction_year','model','feature_group','spatial_fold','buffer_km','single_class_fallback','warnings']})
    (LOG/'model_warnings.json').write_text(json.dumps(warnings,indent=2))
if __name__=='__main__':main()
