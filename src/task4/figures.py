from config import *
import os
os.environ.setdefault('MPLCONFIGDIR',str(LOG/'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COL={'E0':'#984ea3','S0':'#4daf4a','M0':'#555555','M1':'#2286b6'}
def save(fig,name):fig.savefig(FIG/name,dpi=180,bbox_inches='tight');plt.close(fig)
def main():
    a=pd.read_csv(O/'decomposition_results_by_year.csv');ag=pd.read_csv(O/'decomposition_gains_by_year.csv');b=pd.read_csv(O/'spatial_temporal_gains_by_year.csv');c=pd.read_csv(O/'climate_decomposition_gains_by_year.csv');cs=pd.read_csv(O/'climate_spatial_gains_by_year.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,model in zip(axs,['Logistic','RF']):
        for group in ['E0','S0','M0','M1']:
            d=a[(a.model==model)&(a.feature_group==group)].sort_values('prediction_year');ax.plot(d.prediction_year,d.PR_AUC,'o-',color=COL[group],label={'E0':'E0 effort only','S0':'S0 spatial only','M0':'M0 spatial + effort','M1':'M1 + climate'}[group])
        ax.plot(d.prediction_year,d.prevalence,'k:',label='Prevalence');ax.set(title=model,xlabel='Held-out year');ax.legend(fontsize=8)
    axs[0].set_ylabel('Average precision');fig.suptitle('Temporal information decomposition — full candidate sets');fig.tight_layout();save(fig,'T4_information_decomposition.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,comp,title in zip(axs,['M0-E0','M0-S0','M1-M0'],['Spatial history beyond effort','Effort beyond spatial history','Climate beyond both']):
        for model,color in [('Logistic','#2286b6'),('RF','#d07125')]:
            d=ag[(ag.model==model)&(ag.comparison==comp)].sort_values('prediction_year');ax.plot(d.prediction_year,d.delta_AP,'o-',label=model,color=color)
        ax.axhline(0,color='black',lw=1);ax.set(title=f'{title}\n{comp}',xlabel='Held-out year');ax.legend(fontsize=8);ax.tick_params(axis='x',rotation=45)
    axs[0].set_ylabel('Paired Δ average precision');fig.tight_layout();save(fig,'T4_added_value_breakdown.png')
    g=pd.read_csv(D/'spatial_fold_geometry.csv');audit=pd.read_csv(O/'spatial_fold_audit.csv');fig,ax=plt.subplots(figsize=(6.8,6.5));colors=plt.get_cmap('tab10').colors
    for k in range(1,6):
        d=g[g.spatial_fold==k];aa=audit[audit.spatial_fold==k].iloc[0];ax.scatter(d.center_lon,d.center_lat,s=3,color=colors[k-1],label=f'Region{k}: {int(aa.positives)} positives',rasterized=True)
    ax.set(xlabel='Longitude',ylabel='Latitude',title='Frozen coordinate-only spatial regions',aspect=1.25);ax.legend(loc='upper right',fontsize=8);fig.tight_layout();save(fig,'T4_spatial_folds.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.3),sharey=True)
    for ax,model in zip(axs,['Logistic','RF']):
        d=ag[(ag.model==model)&(ag.comparison=='M1-M0')].sort_values('prediction_year');ax.plot(d.prediction_year,d.delta_AP,'o-',color='#555555',label='Temporal only')
        for buf,color,style in [(0,'#2286b6','s-'),(25,'#d07125','^--')]:
            d=b[(b.model==model)&(b.buffer_km==buf)].sort_values('prediction_year');ax.plot(d.prediction_year,d.delta_AP,style,color=color,label=f'Spatial + temporal, buffer{buf}km')
        ax.axhline(0,color='black',lw=1);ax.set(title=model,xlabel='Held-out year');ax.legend(fontsize=8)
    axs[0].set_ylabel('Climate gain: AP(M1) − AP(M0)');fig.suptitle('Region holdout: predictions concatenated before annual AP');fig.tight_layout();save(fig,'T4_temporal_vs_spatial_generalization.png')
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,comp,title in zip(axs,['CN-C0','CA-C0','CNA-CN'],['Normals beyond baseline','Anomalies beyond baseline','Anomalies beyond normals']):
        for model,color in [('Logistic','#2286b6'),('RF','#d07125')]:
            d=c[(c.model==model)&(c.comparison==comp)].sort_values('prediction_year');ax.plot(d.prediction_year,d.delta_AP,'o-',label=model,color=color)
        ax.axhline(0,color='black',lw=1);ax.set(title=f'{title}\n{comp}',xlabel='Held-out year');ax.legend(fontsize=8);ax.tick_params(axis='x',rotation=45)
    axs[0].set_ylabel('Paired Δ average precision');fig.tight_layout();save(fig,'T4_climate_normal_vs_anomaly.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4.3),sharey=True)
    for ax,model in zip(axs,['Logistic','RF']):
        for dataset,label,color in [(c,'Temporal','#555555'),(cs,'Spatial + temporal','#2286b6')]:
            d=dataset[(dataset.model==model)&(dataset.comparison=='CNA-CN')].sort_values('prediction_year');ax.plot(d.prediction_year,d.delta_AP,'o-',color=color,label=label)
        ax.axhline(0,color='black',lw=1);ax.set(title=model,xlabel='Held-out year');ax.legend()
    axs[0].set_ylabel('Dynamic value: AP(CNA) − AP(CN)');fig.suptitle('Do prior-year anomalies add beyond fixed climatic geography?');fig.tight_layout();save(fig,'T4_dynamic_climate_gain_by_year.png')
if __name__=='__main__':main()
