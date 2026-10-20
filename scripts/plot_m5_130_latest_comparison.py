"""Plot the completed current M5 130-week batches, preserving source identities."""
import csv
import json
from pathlib import Path
if __package__ in (None, ""):
 from _results import select_run
else:
 from ._results import select_run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures'
RECORDS = [('Smooth','FOODS_1_002_WI_1_evaluation'),('Erratic','FOODS_1_019_CA_3_evaluation'),('Intermittent','FOODS_2_368_TX_2_evaluation'),('Lumpy','FOODS_1_037_TX_2_evaluation')]
LABELS = ['Rule based (s, S)','Gemini: no communication','Gemini: local communication','Gemini: no forecast input']
COLORS = ['#64748b','#2563eb','#059669','#d97706']
rows=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(4,3,figsize=(17,12.2))
fig.subplots_adjust(left=.235,right=.965,top=.91,bottom=.13,hspace=.55,wspace=.27)
for i,(regime,record_id) in enumerate(RECORDS):
 group=[];reference=None
 for label,method in zip(LABELS,['ss','no_communication','adjacent','no_forecast']):
  run,evidence=select_run('m5',method,regime.lower(),record_id,130)
  m=evidence['configuration'];s=run['evaluation_summary']
  assert run['run_status']=='success' and run['completed_periods']==130
  config={k:m.get(k) for k in ['series_id','input_sha256','num_periods','initial_inventories','capacities','lead_times','holding_costs','backlog_costs','fixed_order_costs']}
  if reference is None:reference=config
  else:assert config==reference,(regime,label)
  row={'regime':regime,'record_id':record_id,'method':label,'cost':s['operational']['global_operational_cost'],'bullwhip':s['global_bullwhip'],'tokens':s['llm_resource']['total_tokens'],'source':str(Path(evidence['source_evidence']['original_path']))}
  rows.append(row);group.append(row)
 for j,key in enumerate(['cost','bullwhip','tokens']):
  ax=axes[i,j];vals=[r[key] for r in group];ax.barh(range(4),vals,color=COLORS,height=.64);ax.invert_yaxis();ax.set_yticks(range(4));ax.set_yticklabels(LABELS if j==0 else ['']*4);ax.tick_params(axis='y',length=0);ax.set_axisbelow(True);ax.grid(axis='x',alpha=.18)
  for spine in ['top','right','left']:ax.spines[spine].set_visible(False)
  ax.set_xlim(0,max(vals)*1.25 if max(vals)>0 else 1)
  ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
  if key=='tokens':ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1e6:g}'))
  for y,v in enumerate(vals):
   text=f'{v:,.0f}' if key=='cost' else f'{v:.5f}' if key=='bullwhip' else ('0' if v==0 else f'{v/1e6:.2f}M')
   ax.text(v+ax.get_xlim()[1]*.012,y,text,va='center',fontsize=8)
  if i==0:ax.set_title(['Global operational cost','Global bullwhip','LLM token usage'][j],fontweight='bold',pad=14)
  if i==3:ax.set_xlabel(['Cost units','Ratio','Million tokens'][j])
 axes[i,0].text(-.85,.5,regime,transform=axes[i,0].transAxes,fontweight='bold',va='center')
fig.suptitle('M5 data: four demand regimes - four inventory policies (130 periods)',fontsize=17,fontweight='bold',y=.97)
fig.text(.5,.025,'Latest completed batches; weeks 1-130. Each panel has its own scale.\nGlobal bullwhip = product of four stage ratios. Cost excludes API charges; tokens include failed attempts.\nIntermittent: FOODS_2_368_TX_2_evaluation. Other series identities are recorded in the accompanying data.',ha='center',fontsize=9,color='#555555')
OUT.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']:fig.savefig(OUT/f'm5_130_comparison.{ext}',dpi=220)
GENERATED = ROOT / 'results/generated'
GENERATED.mkdir(parents=True,exist_ok=True)
(GENERATED/'m5_130_latest_four_regimes_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
with (GENERATED/'m5_130_latest_four_regimes_comparison.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))

