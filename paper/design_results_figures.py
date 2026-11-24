"""Publication graphics from frozen results: no fitting, selection or new tests."""
from pathlib import Path
import os
P=Path(__file__).resolve().parent; R=P.parent; O=R/'outputs/task4'
os.environ.setdefault('MPLCONFIGDIR',str(P/'audit/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
INK='#203548'; GRAY='#728291'; TEAL='#087f8c'; ORANGE='#c65b36'; PURPLE='#7956a1'; LIGHT='#edf2f5'
CM=LinearSegmentedColormap.from_list('signed',['#b54a30','#fcfaf6','#087f8c'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#c7d0d6','text.color':INK,'axes.labelcolor':INK,'xtick.color':GRAY,'ytick.color':INK,'pdf.fonttype':42})
def read(n):return pd.read_csv(O/n)
def save(fig,n):
 fig.savefig(P/'figures'/f'{n}.pdf',bbox_inches='tight',pad_inches=.05)
 fig.savefig(P/'figures'/f'{n}.svg',bbox_inches='tight',pad_inches=.05)
 fig.savefig(P/'figures'/f'{n}.png',dpi=260,bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
def title(fig,x,y,letter,text):
 fig.text(x,y,letter,fontsize=9,fontweight='bold',color=TEAL)
 fig.text(x+.026,y,text,fontsize=9,fontweight='bold')
def box(ax,x,y,w,h,text,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01,rounding_size=.025',facecolor=color,edgecolor='none'))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=7.4,color='white',fontweight='bold')
def row(df,model,comp):return df[(df.model==model)&(df.comparison==comp)].iloc[0]
def forest(ax,rs,ys,colors):
 ax.axvline(0,color=GRAY,lw=.8,zorder=1)
 for r,y,col in zip(rs,ys,colors):
  ax.plot([r.CI_low,r.CI_high],[y,y],color=col,lw=2.4,solid_capstyle='round',zorder=2)
  ax.scatter(r.mean_delta_AP,y,s=28,marker='D',facecolor=col,edgecolor='white',lw=.5,zorder=3)
 ax.grid(axis='x',color=LIGHT,lw=.7);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
 ax.set_xlabel('Mean ΔAP · 95% paired-year interval')
def heat(ax,matrix,labels=False):
 im=ax.imshow(matrix,cmap=CM,norm=TwoSlopeNorm(vmin=-.03,vcenter=0,vmax=.03),aspect='auto')
 ax.set_xticks(range(8),[str(y)[2:] for y in range(2018,2026)])
 ax.set_xlabel('Held-out year (20xx)');ax.tick_params(length=0)
 for y in range(matrix.shape[0]):
  for x in range(8):ax.text(x,y,'+' if matrix[y,x]>0 else '−',ha='center',va='center',fontsize=7.4,color='white' if abs(matrix[y,x])>.018 else INK)
 ax.set_xticks(np.arange(-.5,8,1),minor=True);ax.set_yticks(np.arange(-.5,matrix.shape[0],1),minor=True);ax.grid(which='minor',color='white',lw=1.5);ax.tick_params(which='minor',length=0)
 for sp in ax.spines.values():sp.set_visible(False)
 return im

def information():
 a=read('decomposition_results_by_year.csv');s=read('decomposition_summary.csv')
 # Distinct dark hues; avoid low-contrast yellow on white.
 blue,amber,magenta,green = '#0066A6','#B56800','#A32968','#007A4D'
 fig=plt.figure(figsize=(7.1,3.4))
 for center,label in [(.265,'(a)  Logistic regression'),(.765,'(b)  Random forest')]:
  fig.text(center,.95,label,ha='center',fontsize=9,fontweight='normal',color='black')
 for x,model in [(.12,'Logistic'),(.62,'RF')]:
  ax=fig.add_axes([x,.49,.29,.40]);cols=[blue,amber,magenta,green]
  for i,(group,col) in enumerate(zip(['E0','S0','M0','M1'],cols)):
   d=a[(a.model==model)&(a.feature_group==group)].sort_values('prediction_year');v=d.PR_AUC.values
   ax.axhspan(i-.43,i+.43,color=LIGHT if i%2==0 else 'white',zorder=0)
   ax.scatter(v,i+np.linspace(-.19,.19,8),s=17,facecolor=col,alpha=.85,edgecolor='none',zorder=2)
   mean=float(v.mean());ax.scatter(mean,i,s=46,marker='D',color=col,edgecolor='white',lw=.8,zorder=4)
   ax.text(.049,i,f'{mean:.4f}',ha='right',va='center',fontsize=7.5,fontweight='bold',color=col)
  ax.axvline(d.prevalence.mean(),ls=':',color=GRAY,lw=1)
  ax.set(xlim=(0,.041),ylim=(3.55,-.55),xlabel='Average precision',yticks=range(4),yticklabels=['R','S','R+S','R+S+C']);ax.set_xticks([0,.01,.02,.03,.04]);ax.grid(axis='x',color='white');ax.tick_params(axis='y',length=0,pad=7,labelsize=6.5)
  for tick,col in zip(ax.get_yticklabels(),cols):tick.set_color(col)
 for y,label,col in [(.30,'R: Reporting intensity',blue),(.272,'S: Spatial history',amber),(.244,'R+S: Reporting intensity + spatial history (baseline)',magenta),(.216,'R+S+C: Reporting intensity + spatial history + preceding-year climate',green)]:
  fig.text(.12,y,label,fontsize=6.3,color=col)
 fig.text(.12,.155,'●  Eight annual results   ◆  Equal-year mean     ⋮  Mean prevalence reference',fontsize=7.2,color=GRAY)
 save(fig,'fig2_information')

def climate():
 c=read('climate_decomposition_gains_by_year.csv');s=read('climate_decomposition_gain_summary.csv')
 fig=plt.figure(figsize=(7.1,3.35))
 # Equal half-width panel allocations, including labels.
 for center,label in [(.25,'(a)  Year-by-year evidence'),(.75,'(b)  Mean gain and uncertainty')]:
  fig.text(center,.825,label,ha='center',fontsize=9,fontweight='normal',color='black')
 comps=['CN-C0','CA-C0','CNA-CN'];rs=[];mat=[];labels=[]
 for comp in comps:
  for m in ['Logistic','RF']:
   rs.append(row(s,m,comp));mat.append(c[(c.model==m)&(c.comparison==comp)].sort_values('prediction_year').delta_AP.to_numpy());labels.append(m)
 ax=fig.add_axes([.205,.235,.28,.53]);im=heat(ax,np.array(mat));ax.set_yticks(range(6),['LR','RF']*3)
 for i,comp in enumerate(comps):
  fig.text(.015,.6767-i*(.53/3),['R+S → R+S+N','R+S → R+S+A','R+S+N → R+S+N+A'][i],fontsize=6.3,color=GRAY,va='center')
  # Bracket spans both learner rows; positioned between contrast label and LR/RF.
  ax.plot([-.105,-.14,-.14,-.105],[2*i-.34,2*i-.34,2*i+1.34,2*i+1.34],
          transform=ax.get_yaxis_transform(),clip_on=False,color=INK,lw=.9)
  if i<2:ax.axhline(i*2+1.5,color='white',lw=7,zorder=5)
 f=fig.add_axes([.565,.235,.405,.53]);forest(f,rs,np.arange(6),[GRAY,TEAL]*3);f.set(ylim=(5.5,-.5),xlim=(-.011,.021),yticks=range(6),yticklabels=['LR','RF']*3);f.set_xticks([-.01,0,.01,.02]);f.set_xlabel('ΔAP',labelpad=4)
 for i in range(3):
  f.plot([-.075,-.10,-.10,-.075],[2*i-.34,2*i-.34,2*i+1.34,2*i+1.34],
         transform=f.get_yaxis_transform(),clip_on=False,color=INK,lw=.9)
  if i<2:f.axhline(2*i+1.5,color=LIGHT,lw=1,zorder=0)
 for i,r in enumerate(rs):f.text(.0205,i,f'{int(r.years_positive)}/8',ha='right',va='center',fontsize=7,color=GRAY)
 f.text(.98,1.045,'Positive years',transform=f.transAxes,ha='right',fontsize=6.5,color=GRAY)
 cbax=fig.add_axes([.205,.07,.28,.025]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[-.03,0,.03]);cb.ax.tick_params(labelsize=7,length=0);cb.set_label('Annual ΔAP  ·  loss ← 0 → gain',fontsize=7,labelpad=1)
 fig.text(.565,.09,'◆ Mean    ━ 95% paired-year interval',fontsize=6.8,color=GRAY)
 save(fig,'fig3_climate_decomposition')

def transfer():
 a=read('decomposition_gain_summary.csv');b=read('spatial_temporal_gain_summary.csv');ay=read('decomposition_gains_by_year.csv');by=read('spatial_temporal_gains_by_year.csv')
 g=pd.read_csv(R/'data/processed/task4/spatial_fold_geometry.csv');support=read('spatial_fold_audit.csv')
 fig=plt.figure(figsize=(7.1,3.45))
 for center,label in [(.25,'(a)  Coordinate-only folds'),(.75,'(b)  Mean gain attenuation')]:
  fig.text(center,.955,label,ha='center',fontsize=9,fontweight='normal',color='black')
 ax=fig.add_axes([.045,.24,.39,.64]);cols=['#df865b','#d8b064','#4e9c97','#6686a5','#9a8eac']
 for k,col in zip(range(1,6),cols):
  d=g[g.spatial_fold==k];ax.scatter(d.center_lon,d.center_lat,s=.35,c=col,rasterized=True)
  n=int(support.loc[support.spatial_fold==k,'positives'].iloc[0])
  ax.text(d.center_lon.mean(),d.center_lat.mean(),f'{k}\n(n={n})',ha='center',va='center',fontsize=6.7,fontweight='bold',bbox={'boxstyle':'round,pad=.18','facecolor':'white','edgecolor':'none','alpha':.87})
 ax.set_aspect(1.2);ax.axis('off')
 fig.text(.25,.12,'n = positive cell-years\n318 total · 2018–2025',ha='center',va='top',fontsize=7,color=GRAY,linespacing=1.5)
 f=fig.add_axes([.60,.30,.35,.55]);f.axvline(0,color=GRAY,lw=.8,zorder=0)
 for i,m in enumerate(['Logistic','RF']):
  t=row(a,m,'M1-M0').mean_delta_AP;z=b[(b.model==m)&(b.buffer_km==0)].iloc[0].mean_delta_AP;buf=b[(b.model==m)&(b.buffer_km==25)].iloc[0].mean_delta_AP
  f.plot([z,t],[i,i],lw=3,color='#cbd6dd',zorder=1)
  f.scatter(t,i,s=45,color=INK,zorder=3);f.scatter(z,i,s=48,color=TEAL,zorder=4)
  f.scatter(buf,i+.14,s=32,facecolor='white',edgecolor=ORANGE,lw=1.2,zorder=5)
  f.text(t,i-.19,f'{t:+.5f}',ha='center',fontsize=7,color=INK)
  f.text(z,i+.32,f'{z:+.5f}',ha='center',fontsize=7,color=TEAL)
 f.set(xlim=(-.0007,.008),ylim=(1.65,-.55),yticks=[0,1],yticklabels=['Logistic','RF'],xlabel='Mean ΔAP from adding C to R+S');f.set_xticks([0,.004,.008]);f.tick_params(axis='y',length=0);f.grid(axis='x',color=LIGHT)
 for y,label,col in [(.12,'● Temporal',INK),(.078,'● Primary spatial holdout',TEAL),(.036,'○ 25-km buffer sensitivity',ORANGE)]:
  fig.text(.75,y,label,ha='center',va='top',fontsize=6.6,color=col)
 rf_t=row(a,'RF','M1-M0').mean_delta_AP;rf_s=b[(b.model=='RF')&(b.buffer_km==0)].iloc[0].mean_delta_AP
 fig.text(.5,-.05,f'RF: {rf_t:+.5f} → {rf_s:+.5f}   ·   {100*(1-rf_s/rf_t):.0f}% smaller under primary geographic holdout',fontsize=7.2,ha='center',color=TEAL)
 save(fig,'fig4_transfer')
 blue='#1764A0';green='#008577'
 fig=plt.figure(figsize=(3.45,2.75))
 fig.text(.54,.94,'RF: paired annual climate gains',ha='center',fontsize=9,fontweight='normal',color='black')
 h=fig.add_axes([.19,.21,.76,.65])
 t=ay[(ay.model=='RF')&(ay.comparison=='M1-M0')].sort_values('prediction_year');z=by[(by.model=='RF')&(by.buffer_km==0)].sort_values('prediction_year')
 assert t.prediction_year.tolist()==z.prediction_year.tolist()==list(range(2018,2026))
 tv=t.delta_AP.to_numpy();sv=z.delta_AP.to_numpy()
 h.axhspan(-.0065,0,color='#fbf1ed',zorder=0)
 h.axhline(0,color=GRAY,lw=.8,zorder=1)
 # Separate summary boxes from the connected raw annual observations.
 for x,v,col in [(-.17,tv,blue),(1.17,sv,green)]:
  h.boxplot([v],positions=[x],widths=.20,patch_artist=True,showfliers=False,whis=(0,100),
   boxprops=dict(facecolor=col,alpha=.22,edgecolor=col,lw=.9),
   medianprops=dict(color=col,lw=1.8),whiskerprops=dict(color=col,lw=.8),capprops=dict(color=col,lw=.8))
 label_y=sv.copy();order=np.argsort(sv)
 for j in range(1,len(order)):label_y[order[j]]=max(label_y[order[j]],label_y[order[j-1]]+.0016)
 for i,year in enumerate(t.prediction_year):
  h.plot([0,1],[tv[i],sv[i]],color='#a4b1ba',lw=.85,alpha=.8,zorder=2)
  h.scatter([0,1],[tv[i],sv[i]],c=[blue,green],s=19,edgecolor='white',lw=.4,zorder=3)
  h.plot([1.03,1.33],[sv[i],label_y[i]],color='#a4b1ba',lw=.55,zorder=2)
  h.text(1.36,label_y[i],str(year),fontsize=6,va='center',color=INK)
 h.set(xlim=(-.4,1.72),ylim=(-.0065,.022),xticks=[-.08,1.08],xticklabels=['Temporal','Spatial'],ylabel='Annual ΔAP')
 h.set_yticks([-.005,0,.005,.01,.015,.02]);h.tick_params(axis='x',length=0,labelsize=7);h.grid(axis='y',color=LIGHT,lw=.6);h.set_axisbelow(True)
 fig.text(.54,.07,'8 paired years · boxes: IQR and median',ha='center',fontsize=6.4,color=GRAY)
 save(fig,'fig5_paired_years')

def main():
 information();climate();transfer()
 print('Redesigned Figures 2–5 from frozen data; no models fitted.')
if __name__=='__main__':main()
