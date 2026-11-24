"""Reformat frozen Task 1–4 evidence; never fit or select models."""
from pathlib import Path
import os, json, hashlib, sys
FIGURE1_ONLY = "--figure1-only" in sys.argv
R=Path(__file__).resolve().parents[1]; P=R/'paper'; O=R/'outputs/task4'
os.environ.setdefault('MPLCONFIGDIR',str(P/'audit/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
for d in ['figures','generated','audit']: (P/d).mkdir(exist_ok=True)
inputs=[]
def read(name):
    p=O/name;inputs.append(p);return pd.read_csv(p)
def save(fig,name):
    fig.savefig(P/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(P/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
a=read('decomposition_results_by_year.csv'); ag=read('decomposition_gains_by_year.csv'); b=read('spatial_temporal_gains_by_year.csv'); c=read('climate_decomposition_gains_by_year.csv')
s=read('decomposition_summary.csv'); gains=read('decomposition_gain_summary.csv'); cg=read('climate_decomposition_gain_summary.csv'); bg=read('spatial_temporal_gain_summary.csv'); cs=read('climate_decomposition_summary.csv')
# Revised Figures 2–4 are built by the dedicated publication-graphics module.
if not FIGURE1_ONLY:
 with plt.rc_context():
     from design_results_figures import main as redesign_results
     redesign_results()
# First-year illustration only; no future records displayed as historical sources.
grid=pd.read_csv(R/'data/processed/grid_4km.csv'); occ=pd.read_csv(R/'data/processed/occurrences_with_cells.csv'); first=occ.groupby('cell_id').year.min();g=grid.merge(first.rename('first_year'),on='cell_id',how='left')
test=a[(a.model=='RF')&(a.feature_group=='M0')].sort_values('prediction_year')
panel=pd.read_csv(R/'data/processed/task3/features_primary_3yr.csv.gz',usecols=['cell_id','prediction_year','label']); ids=panel.loc[panel.prediction_year==2018,'cell_id'];src=g[g.first_year<2018];cand=g[g.cell_id.isin(ids)];pos=cand[cand.first_year==2018]
fig=plt.figure(figsize=(3.45,3.65))
ax=fig.add_axes([.19,.30,.77,.64])
ax.scatter(g.x/1000,g.y/1000,s=1,c='#e1e1e1',rasterized=True,label='California grid')
ax.scatter(cand.x/1000,cand.y/1000,s=1,c='#adcbe0',rasterized=True,label='2018 candidates')
ax.scatter(src.x/1000,src.y/1000,s=5,c='#32495a',rasterized=True,label='Observed before 2018')
ax.scatter(pos.x/1000,pos.y/1000,s=19,c='#b35806',marker='x',lw=1.05,label='Newly observed in 2018')
ax.set(aspect='equal',xlabel='EPSG:3310 easting (km)',ylabel='Northing (km)',title='First forecast origin: 2018')
# Dedicated legend outside the map, with readable symbols independent of point density.
from matplotlib.lines import Line2D
handles=[Line2D([],[],marker='s',ls='none',color='#d6d6d6',markersize=6,label='California grid'),Line2D([],[],marker='s',ls='none',color='#adcbe0',markersize=6,label='2018 candidate cells'),Line2D([],[],marker='o',ls='none',color='#32495a',markersize=5,label='Recorded before 2018'),Line2D([],[],marker='x',ls='none',color='#b35806',markersize=7,markeredgewidth=1.2,label='First record in 2018 (held-out)')]
fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.08,-.055),fontsize=8,frameon=False,handletextpad=.7,labelspacing=.40,borderaxespad=0)
save(fig,'fig1_design')
if FIGURE1_ONLY:
    print('Updated Figure 1 only; all other figures and numerical outputs preserved.')
    raise SystemExit(0)
DISPLAY={'E0':'R','S0':'S','M0':'R+S','M1':'R+S+C'}
CONTRAST={'M0-E0':'add S to R','M1-M0':'add C to R+S','CN-C0':'add N to R+S','CNA-CN':'add A to R+S+N'}
rows=[]
for model in ['Logistic','RF']:
    for group in ['E0','S0','M0','M1']:
        r=s[(s.model==model)&(s.feature_group==group)].iloc[0]
        rows.append(f'{model} {DISPLAY[group]} & {r.mean_AP:.5f} & {r.median_AP:.5f} & {r.median_AP_over_prevalence:.2f} & {r.median_ROC_AUC:.3f} \\\\')
(P/'generated/table2_rows.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
rows=[]
for df,model,comp,label in [(gains,m,k,f'{m}: {CONTRAST[k]}') for k in ['M0-E0','M1-M0'] for m in ['Logistic','RF']]+[(cg,m,k,f'{m}: {CONTRAST[k]}') for k in ['CN-C0','CNA-CN'] for m in ['Logistic','RF']]+[(bg[bg.buffer_km==0],'RF','M1-M0','RF spatial: add C to R+S')]:
    r=df[(df.model==model)&(df.comparison==comp)].iloc[0]
    rows.append(f'{label} & ${r.mean_delta_AP:+.5f}$ & ${r.median_delta_AP:+.5f}$ & $[{r.CI_low:+.5f}, {r.CI_high:+.5f}]$ & {r.years_positive}/8 \\\\')
(P/'generated/table3_rows.tex').write_text('\n'.join(rows)+'\n'+r'\bottomrule'+'\n')
# Manuscript macros and exact source-bound claim ledger.
records={};macros=[]
def val(key,v,source,selector,fmt='.5f'):
    records[key]={'value':float(v),'source':source,'selector':selector,'formatted':format(v,fmt)};macros.append('\\newcommand{\\'+key+'}{'+format(v,fmt)+'}')
for df,source,prefix,comparisons in [(gains,'decomposition_gain_summary.csv','D',['M0-E0','M0-S0','M1-M0']),(cg,'climate_decomposition_gain_summary.csv','C',['CN-C0','CA-C0','CNA-CN'])]:
    for model,mkey in [('RF','RF'),('Logistic','LR')]:
        for comp in comparisons:
            r=df[(df.model==model)&(df.comparison==comp)].iloc[0];base=prefix+mkey+comp.replace('-','').replace('0','Z').replace('1','O')
            for suffix,col in [('Mean','mean_delta_AP'),('Median','median_delta_AP'),('Low','CI_low'),('High','CI_high')]:val(base+suffix,r[col],source,f'{model}, {comp}: {col}')
for model,mkey in [('RF','RF'),('Logistic','LR')]:
    for group in ['E0','S0','M0','M1']:
        r=s[(s.model==model)&(s.feature_group==group)].iloc[0];base=mkey+group.replace('0','Z').replace('1','O')
        val(base+'AP',r.mean_AP,'decomposition_summary.csv',f'{model}, {group}: mean_AP')
        val(base+'Lift',r.median_AP_over_prevalence,'decomposition_summary.csv',f'{model}, {group}: median_AP_over_prevalence','.2f')
    for group in ['CN','CA','CNA']:
        r=cs[(cs.model==model)&(cs.feature_group==group)].iloc[0];val(mkey+group+'AP',r.mean_AP,'climate_decomposition_summary.csv',f'{model}, {group}: mean_AP')
    for buf in [0,25]:
        r=bg[(bg.model==model)&(bg.buffer_km==buf)].iloc[0];base=mkey+('Spatial' if buf==0 else 'Buffer')
        for suffix,col in [('Mean','mean_delta_AP'),('Low','CI_low'),('High','CI_high')]:val(base+suffix,r[col],'spatial_temporal_gain_summary.csv',f'{model}, buffer={buf}: {col}')
rf=gains[(gains.model=='RF')&(gains.comparison=='M1-M0')].iloc[0];bs=bg[(bg.model=='RF')&(bg.buffer_km==0)].iloc[0];r0=s[(s.model=='RF')&(s.feature_group=='M0')].iloc[0]
val('RelativeClimateGain',100*rf.mean_delta_AP/r0.mean_AP,'decomposition_summary.csv + decomposition_gain_summary.csv','100 * mean delta / M0 mean AP','.1f')
val('ClimateAttenuation',100*(1-bs.mean_delta_AP/rf.mean_delta_AP),'spatial_temporal_gain_summary.csv + decomposition_gain_summary.csv','100*(1-spatial/temporal mean delta)','.1f')
val('Candidates',test.n_test.sum(),'decomposition_results_by_year.csv','RF M0; sum n_test',',.0f');val('Positives',test.n_positive_test.sum(),'decomposition_results_by_year.csv','RF M0; sum n_positive_test','.0f')
(P/'generated/numbers.tex').write_text('\n'.join(macros)+'\n');(P/'audit/numerical_claims.json').write_text(json.dumps(records,indent=2))
paths=inputs+[R/'data/processed/grid_4km.csv',R/'data/processed/occurrences_with_cells.csv',R/'data/processed/task3/features_primary_3yr.csv.gz']
(P/'audit/asset_sources.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
print('Saved four figures, two table bodies, numerical macros and source hashes. No model fitting.')
