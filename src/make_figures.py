from common import *
import os
os.environ["MPLCONFIGDIR"]=str(ROOT/"logs/matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    cells=pd.read_csv(ROOT/'data/processed/first_observation_cells.csv');g,p=boundary()
    fig,ax=plt.subplots(figsize=(7,8))
    for poly in p.geoms:
        x,y=poly.exterior.xy;ax.plot(np.array(x)/1000,np.array(y)/1000,color='#888888',lw=.6)
    sub=cells.loc[cells.first_observation_year.le(2025)]
    sc=ax.scatter(sub.x/1000,sub.y/1000,c=sub.first_observation_year,cmap='viridis',s=9,marker='s',vmin=1900,vmax=2025,linewidths=0)
    fig.colorbar(sc,ax=ax,shrink=.7,label='First observation year in downloaded records')
    ax.set(aspect='equal',xlabel='California Albers x (km)',ylabel='California Albers y (km)',title='Salsola tragus: first observed 4-km cells\nComplete years through 2025; not biological invasion dates')
    fig.tight_layout();fig.savefig(ROOT/'figures/F1_salsola_first_observation_map.png');plt.close(fig)
    a=pd.read_csv(ROOT/'outputs/observations_by_year.csv');a=a.loc[a.year.between(2000,2025)]
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    axes[0].bar(a.year,a.total_observations,color='#267c9b');axes[0].set(ylabel='Valid observations',title='Temporal support after cleaning; partial 2026 excluded')
    axes[1].bar(a.year,a.newly_observed_cells,color='#d78d32',label='Newly observed 4-km cells')
    axes[1].axhline(20,color='#666666',ls='--',label='20/year heuristic');axes[1].axhline(30,color='#999999',ls=':',label='30/year heuristic')
    axes[1].set(xlabel='Observation year',ylabel='New cells');axes[1].legend();axes[1].set_xticks(a.year);axes[1].tick_params(axis='x',rotation=60)
    fig.tight_layout();fig.savefig(ROOT/'figures/F2_observations_by_year.png');fig.savefig(ROOT/'figures/observations_by_year.png');plt.close(fig)
    c=pd.read_csv(ROOT/'outputs/candidate_counts_by_year.csv');fig,ax=plt.subplots(figsize=(10,5))
    for label,sub in c.loc[c.year.ge(2017)].groupby('candidate_set'):ax.plot(sub.year,sub.positives,marker='o',label=label)
    ax.axhline(20,color='gray',ls='--');ax.set(xlabel='Target year',ylabel='Newly observed candidate cells',title='Candidate neighborhood sensitivity (historical sources only)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/F3_candidate_support.png');plt.close(fig)
if __name__=='__main__':main()
