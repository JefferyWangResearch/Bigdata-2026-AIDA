from setup import *
import os
os.environ.setdefault("MPLCONFIGDIR",str(LOG/"matplotlib"))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.calibration import calibration_curve
COL={'M0':'#555555','M1':'#2286b6','M2':'#d07125'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save(fig,name):fig.savefig(FIG/name,dpi=180,bbox_inches='tight');plt.close(fig)
def main():
    r=pd.read_csv(O/'model_results_by_year.csv');p=pd.read_csv(O/'paired_feature_gain_by_year.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,model in zip(axs,['Logistic','RF']):
        for group in GROUP_ORDER:
            a=r[(r.model==model)&(r.feature_group==group)].sort_values('prediction_year');ax.plot(a.prediction_year,a.PR_AUC,'o-',label={'M0':'M0 spatial + effort','M1':'M1 + climate','M2':'M2 + extremes'}[group],color=COL[group])
        ax.plot(a.prediction_year,a.positive_rate,':',color='black',label='Prevalence');ax.set(title=model,xlabel='Held-out prediction year');ax.legend(fontsize=8)
    axs[0].set_ylabel('PR-AUC (average precision)');fig.suptitle('Forward forecasts on complete 50-km candidate sets');fig.tight_layout();save(fig,'T3_model_ablation_prauc.png')
    for comparison,fname,title in [('M1-M0','T3_climate_gain_by_year.png','Ordinary climate added value'),('M2-M1','T3_extreme_gain_by_year.png','Climate extremes added value')]:
        fig,ax=plt.subplots(figsize=(9,4));years=sorted(p.prediction_year.unique());xx=np.arange(len(years))
        for offset,model,color in [(-.18,'Logistic','#2286b6'),(.18,'RF','#d07125')]:
            a=p[(p.model==model)&(p.comparison==comparison)].sort_values('prediction_year');ax.bar(xx+offset,a.delta_PR_AUC,width=.34,label=model,color=color)
        ax.axhline(0,color='black',lw=1);ax.set(xticks=xx,xticklabels=years,ylabel=f'Δ average precision ({comparison})',xlabel='Held-out prediction year',title=title);ax.legend();fig.tight_layout();save(fig,fname)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    calibration=[]
    for ax,model in zip(axs,['Logistic','RF']):
        maxval=0
        for group in GROUP_ORDER:
            a=pd.concat([pd.read_csv(O/f'primary_3yr/{y}_{model}_{group}_predictions.csv.gz') for y in range(2018,2026)])
            # Equal-count risk bins; tie handling preserves complete predictions.
            a['bin']=pd.qcut(a.probability,10,duplicates='drop')
            bins=a.groupby('bin',observed=True).agg(mean_predicted=('probability','mean'),observed_rate=('label','mean'),n=('label','size'),positives=('label','sum')).reset_index(drop=True);bins['model']=model;bins['feature_group']=group;calibration.append(bins)
            ax.plot(bins.mean_predicted,bins.observed_rate,'o-',label=group,color=COL[group]);maxval=max(maxval,bins.mean_predicted.max(),bins.observed_rate.max())
        ax.plot([0,maxval],[0,maxval],'k--',lw=1);ax.set(title=model,xlabel='Mean corrected predicted probability',ylabel='Observed new-cell frequency');ax.legend()
    pd.concat(calibration).to_csv(O/'calibration_bins.csv',index=False);fig.suptitle('Pooled held-out calibration, 2018–2025 (quantile bins)');fig.tight_layout();save(fig,'T3_calibration.png')
    grid=pd.read_csv(R/'data/processed/grid_4km.csv');first=pd.read_csv(R/'data/processed/first_observation_cells.csv');hist=first[first.first_observation_year<=2024]
    fig,axs=plt.subplots(1,3,figsize=(12,5.1),sharex=True,sharey=True)
    for ax,group in zip(axs,GROUP_ORDER):
        a=pd.read_csv(O/f'primary_3yr/2025_RF_{group}_predictions.csv.gz');ax.scatter(grid.center_lon,grid.center_lat,s=1,color='#ededed',rasterized=True)
        sc=ax.scatter(a.center_lon,a.center_lat,c=np.clip(a.probability,1e-5,.05),norm=LogNorm(vmin=1e-5,vmax=.05),s=3,cmap='viridis',rasterized=True)
        ax.scatter(hist.center_lon,hist.center_lat,s=3,color='black',alpha=.35,label='Observed through 2024');pos=a[a.label==1];ax.scatter(pos.center_lon,pos.center_lat,s=28,facecolors='none',edgecolors='#e54838',linewidths=.8,label='Newly observed 2025');ax.set(title=f'2025 RF {group}',xlabel='Longitude',aspect=1.25);ax.legend(loc='upper right',fontsize=6)
    axs[0].set_ylabel('Latitude');fig.subplots_adjust(right=.88,wspace=.08);cax=fig.add_axes([.90,.2,.018,.6]);fig.colorbar(sc,cax=cax,label='Corrected predicted score (log scale)');fig.suptitle('Preselected 2025 maps — observational risk, calibration not assumed');save(fig,'T3_2025_expansion_risk_map.png')

def robustness():
    variants=[('Primary50km',''),('Strict coordinates','_strict_3yr'),('Previous-year effort','_primary_1yr'),('Radius25km','_radius25'),('Radius100km','_radius100')]
    if not (O/'incremental_gain_summary_radius100.csv').exists():return
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,comp,title in zip(axs,['M1-M0','M2-M1'],['Ordinary climate increment','Extreme-feature increment']):
        for i,(label,suffix) in enumerate(variants):
            d=pd.read_csv(O/f'incremental_gain_summary{suffix}.csv')
            for model,offset,color in [('Logistic',-.14,'#2286b6'),('RF',.14,'#d07125')]:
                a=d[(d.model==model)&(d.comparison==comp)].iloc[0];ax.errorbar(a.mean_delta_PR_AUC,i+offset,xerr=[[a.mean_delta_PR_AUC-a.CI_low],[a.CI_high-a.mean_delta_PR_AUC]],fmt='o',capsize=3,color=color,label=model if i==0 else None)
        ax.axvline(0,color='black',lw=1);ax.set(title=title,xlabel=f'Mean paired AP gain ({comp}), 95% year-bootstrap CI',yticks=range(5),yticklabels=[x[0] for x in variants]);ax.invert_yaxis();ax.legend(fontsize=8)
    # Shared y axes must only be inverted once.
    axs[0].set_ylim(4.6,-.6)
    fig.suptitle('General climate is more consistent; extreme gains depend on data/model choices');fig.tight_layout();save(fig,'T3_robustness_summary.png')

GROUP_ORDER=['M0','M1','M2']
if __name__=='__main__':
    main();robustness()
