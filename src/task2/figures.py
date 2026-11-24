from setup import *
import os
os.environ['MPLCONFIGDIR']=str(R/'logs/task2/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from exposure import window_data,GEO

def main():
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    pairs=pd.read_csv(D/'matched_pairs_primary.csv');years=pd.read_csv(R/'outputs/task2_alignment_by_year.csv');null=pd.read_csv(R/'outputs/wind_permutation_results.csv');s=pd.read_csv(R/'outputs/task2_sensitivity_results.csv');primary=s.loc[s.coordinate_quality_definition.eq('primary')&s.radius.eq(50)&s.L.eq(25)&s.wind_window.eq('Sep-Jan')].iloc[0]
    p=pairs.drop_duplicates(['year','positive_cell']);fig,ax=plt.subplots(figsize=(8,5));ax.hist(p.positive_alignment,bins=np.linspace(-1,1,41),density=True,alpha=.6,label=f'Newly observed ({len(p)})',color='#cd8b37');ax.hist(pairs.negative_alignment,bins=np.linspace(-1,1,41),density=True,alpha=.5,label='Effort/spatial matched controls (with reuse)',color='#267c9b');ax.set(xlabel='Distance-weighted signed directional alignment',ylabel='Density',title='Primary: 50km / L25km / completed Sep–Jan');ax.legend();fig.tight_layout();fig.savefig(R/'figures/T2_alignment_positive_vs_negative.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.vlines(years.year,years.ci_low,years.ci_high,color='#267c9b');ax.scatter(years.year,years.delta_alignment,color='#267c9b');ax.axhline(0,color='gray',ls='--');ax.set(xlabel='First-observation target year',ylabel='Positive median minus matched-negative median',title='Year-specific alignment differences\n95% matched-set bootstrap intervals; spatial dependence remains');ax.set_xticks(years.year);fig.tight_layout();fig.savefig(R/'figures/T2_alignment_difference_by_year.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.hist(null.statistic,bins=30,color='#b9cbd4',label='500 coherent field rotations');ax.axvline(primary.observed_effect,color='#ce7923',lw=2,label='Real direction');ax.axvline(primary.reversed_effect,color='#733c93',lw=2,ls='--',label='Reversed direction');ax.set(xlabel='Equal-year mean of median alignment differences',ylabel='Permutation count',title=f'Actual vs direction controls; one-sided p={primary.permutation_p:.3f}');ax.legend();fig.tight_layout();fig.savefig(R/'figures/T2_real_shuffled_reversed.png');plt.close(fig)
    year=2019;sources=pd.read_csv(D/'historical_sources.csv');sources=sources.loc[sources.quality.eq('primary')&sources.year.eq(year)];grid=pd.read_csv(R/'data/processed/grid_4km.csv').set_index('cell_id');pp=pairs.loc[pairs.year.eq(year)];a=grid.loc[pp.positive_cell.unique()];b=grid.loc[pp.negative_cell.unique()];g,pr=boundary()
    fig,ax=plt.subplots(figsize=(9,7))
    for poly in pr.geoms:
        x,y=poly.exterior.xy;ax.plot(np.array(x)/1000,np.array(y)/1000,color='gray',lw=.6)
    ax.scatter(sources.x/1000,sources.y/1000,s=9,c='gray',label='Historical source cells')
    ax.scatter(b.x/1000,b.y/1000,s=20,marker='x',color='#267c9b',label='Matched negatives');ax.scatter(a.x/1000,a.y/1000,s=45,facecolors='none',edgecolors='#d47725',lw=1.5,label='Newly observed cells')
    ids,speed,fr,dates=window_data(year,'Sep-Jan');u=np.nanmean(-speed*np.sin(np.deg2rad(fr)),axis=0);v=np.nanmean(-speed*np.cos(np.deg2rad(fr)),axis=0);idx={c:k for k,c in enumerate(ids)}
    ss=sources.loc[sources.center_lat.between(32,36.8)&sources.center_lon.between(-121.5,-115)].iloc[::12];j=[idx[c] for c in ss.cell_id];bearing=np.rad2deg(np.arctan2(u[j],v[j]));lo,la,_=GEO.fwd(ss.center_lon.values,ss.center_lat.values,bearing,np.sqrt(u[j]**2+v[j]**2)*8000);xx,yy=FWD.transform(lo,la);ax.quiver(ss.x/1000,ss.y/1000,(xx-ss.x)/1000,(yy-ss.y)/1000,angles='xy',scale_units='xy',scale=1,color='#296a3a',width=.004)
    xx,yy=FWD.transform([-121.5,-115],[32,36.8]);ax.set_xlim(min(xx)/1000,max(xx)/1000);ax.set_ylim(min(yy)/1000,max(yy)/1000);ax.set(aspect='equal',xlabel='California Albers x (km)',ylabel='California Albers y (km)',title='Illustrative Southern California view, target 2019\nArrows: transport during Sep 2017–Jan 2018, 8km per m/s');ax.legend(loc='lower left',fontsize=8);fig.tight_layout();fig.savefig(R/'figures/T2_example_directional_spread_map.png');plt.close(fig)
    effort=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');fig,axes=plt.subplots(2,1,figsize=(9,7))
    e=effort.groupby('year').all_plant_observation_count.sum();axes[0].plot(e.index,e.values,marker='o');axes[0].set(ylabel='Vascular plant records',title=f'Observation effort within fixed {effort.cell_id.nunique():,}-cell query set\nTargeted candidate pool, not statewide census')
    ep=p.groupby('year').positive_effort.median();en=pairs.groupby('year').negative_effort.median();axes[1].plot(ep.index,ep.values,marker='o',label='Positive cells');axes[1].plot(en.index,en.values,marker='s',label='Matched negatives');axes[1].set(xlabel='Target year',ylabel='Median preceding 3-year annual count');axes[1].legend();fig.tight_layout();fig.savefig(R/'figures/T2_observation_effort.png');plt.close(fig)
if __name__=='__main__':main()
