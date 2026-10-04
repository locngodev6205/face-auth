"""Rebuild figures from existing immutable run artifacts; no model execution."""
from pathlib import Path
import csv, json, sys, os, platform
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mast_pad_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
BASE = ROOT / 'antispoof/notebooks/final/binary_crossdomain_v2'
D17 = BASE / 'output/17_scale100k_C_P3SF_R7SC_crop15'
D18 = BASE / 'output/18_fullscale_C_P3SF_R7SC_crop15'
LABELS = ['Clean', 'WBST', 'MAST-PAD']
COLORS = ['#78838f', '#397ab7', '#008b83']
IDS17 = ['C_100K_crop15', 'P3_SF_100K_crop15', 'R7_SC_100K_crop15']
IDS18 = ['C_FULL_crop15', 'P3_SF_FULL_crop15', 'R7_SC_FULL_crop15']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':14, 'axes.titlesize':18,
                    'axes.labelsize':15, 'xtick.labelsize':13, 'ytick.labelsize':13,
                    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
                    'axes.spines.top':False, 'axes.spines.right':False,
                    'figure.facecolor':'white', 'savefig.facecolor':'white'})
records = {}

def read(p): return json.loads(p.read_text())
def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))
def rel(p): return str(p.relative_to(ROOT))
def save(fig, name, sources, data, caption, note='', svg=False, static=False):
    files = []
    for ext in ['png','pdf'] + (['svg'] if svg else []):
        f = OUT / f'{name}.{ext}'
        fig.savefig(f, dpi=300, bbox_inches='tight', pad_inches=.18)
        files.append(f.name)
    plt.close(fig)
    records[name] = dict(files=files, sources=[rel(p) for p in sources], data=data,
                         caption=caption, note=note, static_fallback=False,
                         supplied_context=static)

def polish(ax):
    ax.set_axisbelow(True); ax.grid(axis='y', color='#e7ecef', linewidth=.8)

def footer(fig, text): fig.text(.5,.025,text,ha='center',va='bottom',fontsize=11,color='#4e5d68')

hist = [rows(D17/'runs'/r/'training_history.csv') for r in IDS17]
best = [read(D17/'runs'/r/'completion.json')['best_epoch'] for r in IDS17]
sources = [D17/'runs'/r/'metrics/lcc_combined_metrics.json' for r in IDS17]
auc = [read(p)['auc'] for p in sources]
delta = (auc[2]-auc[0])*100
fig, ax = plt.subplots(figsize=(12,6.5))
fig.subplots_adjust(top=.77,bottom=.18)
fig.suptitle('Main results: pooled LCC',fontsize=22,y=.98)
fig.text(.5,.905,'Main experiment; source threshold frozen before LCC',ha='center',fontsize=13)
ax.bar(LABELS,auc,color=COLORS,width=.55)
ax.set_ylim(0,1); ax.set_ylabel('LCC Combined AUC'); polish(ax)
for i,y in enumerate(auc): ax.text(i,y+.014,f'{y:.3f}',ha='center',fontweight='bold',fontsize=17)
ax.annotate('', xy=(2,.958),xytext=(0,.958),arrowprops=dict(arrowstyle='->',color=COLORS[2],lw=2))
ax.text(1,.971,f'+{delta:.2f} pp',ha='center',color=COLORS[2],fontweight='bold',fontsize=16)
footer(fig,'One training seed; LCC Combined pools training, development and evaluation splits.')
save(fig,'fig_main_lcc_combined_auc',sources,dict(zip(LABELS,auc))|{'gain_pp':delta},
     'MAST-PAD improves pooled LCC AUC by 1.96 percentage points over Clean in the main experiment.')

complexity = D18/'complexity_report.json'
params = read(complexity)['shared_clean_inference']['params_m']
context = {'Graph for Transformer Feature':{'params_M':10.84,'auc':.833},
           'AENet':{'params_M':11.22,'auc':.868},
           'MAST-PAD':{'params_M':params,'auc':auc[2]}}
fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(bottom=.2,top=.86,right=.85)
ax.set_title('Compact deployment: contextual LCC comparison',pad=18)
for name,v in context.items():
    color=COLORS[2] if name=='MAST-PAD' else '#7a91ad'
    ax.scatter(v['params_M'],v['auc'],s=220 if name=='MAST-PAD' else 110,color=color,zorder=3)
ax.annotate('MAST-PAD\n0.434M params, AUC 0.868',(params,auc[2]),xytext=(15,15),textcoords='offset points',fontsize=15,color=COLORS[2],fontweight='bold')
ax.annotate('AENet',(11.22,.868),xytext=(-10,16),ha='right',textcoords='offset points',fontsize=15)
ax.annotate('Graph for Transformer Feature',(10.84,.833),xytext=(-15,-26),ha='right',textcoords='offset points',fontsize=14)
ax.set_xscale('log');ax.set_xlim(.25,22);ax.set_ylim(.82,.885)
ax.set_xticks([.3,.5,1,2,5,10,20]);ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('Parameters (M), log scale');ax.set_ylabel('LCC AUC');polish(ax)
footer(fig,'Contextual references; protocols differ.')
save(fig,'fig_lcc_auc_vs_params_context',sources[-1:]+[complexity,ROOT/'md/mast_pad_codex_figure_prompts.md'],context,
     'MAST-PAD has 434,434 deployment parameters and 0.868 pooled LCC AUC. Published reference points are contextual comparisons.',
     'Reference numbers are supplied by the figure prompt, not independently reproduced measurements. The MAST-PAD AUC uses unrounded run JSON.',static=True)

fig,ax=plt.subplots(figsize=(16,9));ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off')
ax.text(.5,8.55,'MAST-PAD: margin-aware spectral supervision',fontsize=24,fontweight='bold',color='#173646')
ax.text(.5,8,'Training: choose the spectral view that most degrades the label-aligned margin',fontsize=15,color='#526773')
def box(x,y,w,h,text,color='#eaf5f4',size=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.15',facecolor=color,edgecolor='#38877f',lw=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#173646')
def arrow(a,b,color='#38877f',style='arc3',lw=1.6):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color=color,connectionstyle=style,lw=lw))
box(.5,4.35,1.6,1.15,'Clean\ncrop')
for y,t in [(6.2,'LOW'),(4.55,'MID'),(2.9,'HIGH')]:
    box(2.9,y,2.15,1.1,t+'\nattenuation',size=14)
    arrow((2.2,4.92),(2.78,y+.55));arrow((5.17,y+.55),(6.05,4.92))
box(6.2,4.3,2.6,1.25,'Shared\nMiniFASNetV2 PAD',color='#eaf1fb',size=15)
# Clean view enters the same frozen selection forward as the spectral candidates.
arrow((1.3,5.62),(1.3,7.55));arrow((1.3,7.55),(7.5,7.55));arrow((7.5,7.55),(7.5,5.7))
ax.text(4.3,7.72,'Clean view: same model',ha='center',fontsize=12,color='#526773')
box(9.55,4.3,3.1,1.25,'Label-aligned margins\n'+r'$m_{low},\ m_{mid},\ m_{high}$',size=14)
arrow((8.95,4.92),(9.4,4.92))
box(13.35,4.3,2.05,1.25,'Worst view\n'+r'$k^*=\arg\min_k m_k$',size=14)
arrow((12.8,4.92),(13.2,4.92))
box(10.45,2.05,4.9,1.15,'Harmful only if\n'+r'$m_{worst}<m_{clean}$',size=17)
arrow((14.38,4.16),(14.38,3.35))
arrow((8.05,4.16),(10.3,2.8));ax.text(8.6,3.3,r'$m_{clean}$',fontsize=15)
box(.5,.65,8.2,1.0,r'$L=0.75\,CE_{clean}+0.25\,CE_{harmful}+L_{aux}$',size=19)
arrow((12.85,1.91),(8.84,1.18))
ax.text(.65,2.4,'Spectral views are training-only',fontsize=15,color=COLORS[2],fontweight='bold')
ax.text(.65,1.98,'Warmup: clean only. No harmful sample: spectral loss is zero.',fontsize=12,color='#526773')
ax.text(.5,.03,'Inference: crop1.5 → 80×80 → MiniFASNetV2 → Real / Attack',fontsize=16,color='#173646')
save(fig,'fig_mast_pad_method_flow',[BASE/'17_scale100k_C_P3SF_R7SC_crop15.ipynb',D17/'runs'/IDS17[2]/'config.json'],
     {'active_loss_weights':[.75,.25],'crop_factor':1.5,'input_size':[80,80],
      'selector':'argmin label-aligned margin','gate':'worst margin < clean margin','warmup_epochs':1},
     'MAST-PAD selects the minimum-margin spectral view and applies harmful-view supervision when its margin is below the clean margin. Deployment uses the clean binary spatial model.',
     'Selection uses eval/no_grad (BN-safe). Auxiliary heads are training-only; harmful CE is averaged over harmful samples.',svg=True)

sp = D17/'runs'/IDS17[2]/'spectral_diagnostics.csv'
spec=rows(sp);epochs=np.array([int(r['epoch']) for r in spec]);last=spec[-1]
curves={k:[float(r[k]) for r in spec] for k in ['harmful_fraction','margin_drop','spectral_flip_rate']}
fig,(ax,ab)=plt.subplots(1,2,figsize=(15,6.8),gridspec_kw={'width_ratios':[1.7,1]})
fig.subplots_adjust(bottom=.25,top=.83,wspace=.48)
fig.suptitle('MAST-PAD: what makes a spectral view harmful?',fontsize=22)
ax.plot(epochs,curves['harmful_fraction'],color=COLORS[2],label='Harmful fraction',lw=2.5)
ax.plot(epochs,curves['spectral_flip_rate'],color='#9666a5',label='Prediction flip rate',lw=2.5)
ax.set_ylim(0,1);ax.set_ylabel('Sample fraction');ax.set_xlabel('Epoch');ax.set_title('A  Active training diagnostics',loc='left',fontsize=17);polish(ax)
at=ax.twinx();at.spines['right'].set_visible(True)
at.plot(epochs,curves['margin_drop'],color=COLORS[1],label='Mean margin drop',lw=2.5,ls='--')
at.set_ylim(0,.7);at.set_ylabel('Mean margin drop (logit units)',color=COLORS[1])
ax.axvline(best[2],color='#55616e',ls='--',lw=1.2);ax.axvline(int(last['epoch']),color='#55616e',ls=':',lw=1.2)
ax.text(best[2]+.15,.95,f'Best: {best[2]}',fontsize=11);ax.text(int(last['epoch'])-.15,.9,f'Stop: {last["epoch"]}',ha='right',fontsize=11)
lines=ax.lines[:2]+at.lines;ax.legend(lines,[l.get_label() for l in lines],loc='lower left',fontsize=11)
ax.set_xticks([2,5,8,11,14,17])
ax.annotate(f'Final flip rate: {curves["spectral_flip_rate"][-1]*100:.3f}%',
            (epochs[-1],curves['spectral_flip_rate'][-1]), xytext=(-8,21),
            textcoords='offset points',ha='right',fontsize=11,color='#9666a5')
fractions=[float(last[k+'_fraction']) for k in ['low','mid','high']]
ab.bar(['LOW','MID','HIGH'],np.array(fractions)*100,color=['#9bc6df',COLORS[1],COLORS[2]],width=.65)
ab.set_ylim(0,52);ab.set_ylabel('Selected band (%)');ab.set_title(f'B  Band selection at epoch {last["epoch"]}',loc='left',fontsize=16);polish(ab)
for i,v in enumerate(fractions):ab.text(i,v*100+1,f'{v*100:.2f}%',ha='center',fontsize=14,fontweight='bold')
fig.text(.5,.105,'MAST-PAD uses margin degradation, not only prediction flips,\nto select harmful spectral supervision.',ha='center',fontsize=14)
footer(fig,'Active epochs only; epoch 1 is clean warmup. Left and right axes use different units.')
save(fig,'fig_mast_pad_training_diagnostics',[sp,D17/'runs'/IDS17[2]/'completion.json'],
     {'epochs':epochs.tolist(),'curves':curves,'last_epoch':int(last['epoch']),
      'last_band_fractions':dict(zip(['LOW','MID','HIGH'],fractions)), 'best_epoch':best[2]},
     'MAST-PAD uses margin degradation, not only prediction flips, to select harmful spectral supervision.',
     'Spectral CSV contains all active epochs 2–17; no static fallback used. Band selection counts all samples, not just harmful samples. Spectral flip rate counts clean-correct / selected-view-wrong samples divided by total samples (not every label change).')

fig,axes=plt.subplots(1,2,figsize=(15,6.4));fig.subplots_adjust(bottom=.24,top=.82,wspace=.24)
fig.suptitle('Main experiment: validation curves',fontsize=22)
curve_data={}
for label,color,history,b in zip(LABELS,COLORS,hist,best):
    e=[int(r['epoch']) for r in history]
    vauc=[float(r['val_auc']) for r in history];acer=[float(r['val_acer'])*100 for r in history]
    curve_data[label]={'epochs':e,'val_auc':vauc,'val_acer_percent':acer,'best_epoch':b,
                       'checkpoint_eligible':[r['eligible_for_final_checkpoint'] for r in history]}
    for ax,values in zip(axes,[vauc,acer]):
        ax.plot(e,values,color=color,lw=2,marker='o',ms=4,label=label)
        ax.scatter(b,values[e.index(b)],s=210,marker='*',color=color,edgecolors='#223b48',lw=.6,zorder=4)
for ax,title,yl in zip(axes,['A  Validation AUC','B  Validation ACER'],['Validation AUC','Validation ACER (%)']):
    ax.set_title(title,loc='left');ax.set_xlabel('Epoch');ax.set_ylabel(yl);polish(ax)
    ax.set_xticks([1,3,6,9,12,15,18]);ax.axvspan(.7,1.3,color='#ecf1f4',zorder=0)
    for m in [6,14]:ax.axvline(m,color='#bbc5cd',ls=':',lw=1)
axes[0].legend(loc='lower right',fontsize=12)
fig.text(.5,.105,'★ Selected checkpoints: Clean 15 · WBST 15 · MAST-PAD 11',ha='center',fontsize=14)
footer(fig,'Epoch 1: clean warmup for WBST / MAST-PAD (ineligible); Clean is eligible. Dotted lines: LR milestones 6 and 14.')
save(fig,'fig_training_curves_val_auc_acer',[D17/'runs'/r/'training_history.csv' for r in IDS17]+[D17/'runs'/r/'completion.json' for r in IDS17],curve_data,
     'Source validation metrics are evaluated throughout training. Stars mark selected checkpoints, including the original warmup eligibility rules.',
     'AUC and ACER axes are scaled to the observed validation range; all recorded epochs are shown. Best selection is preserved, not chosen using LCC.')

full_sources=[D18/'runs'/r/'metrics'/f for r in IDS18 for f in ['lcc_evaluation_metrics.json','lcc_combined_metrics.json']]
full_data={label:{'LCC Evaluation AUC':read(D18/'runs'/r/'metrics/lcc_evaluation_metrics.json')['auc'],
                  'LCC Combined AUC':read(D18/'runs'/r/'metrics/lcc_combined_metrics.json')['auc']} for label,r in zip(LABELS,IDS18)}
fig,ax=plt.subplots(figsize=(12,6.8));fig.subplots_adjust(bottom=.24,top=.78)
fig.suptitle('Full-scale: secondary scale-up evidence',fontsize=22,y=.97)
fig.text(.5,.885,'Secondary full-scale check; the main model is MAST-PAD.',ha='center',fontsize=12)
for i,(label,color) in enumerate(zip(LABELS,COLORS)):
    v=list(full_data[label].values());xs=np.arange(2)+(i-1)*.23
    ax.bar(xs,v,width=.22,color=color,label=label)
    for x,y in zip(xs,v):ax.text(x,y+.015,f'{y:.3f}',ha='center',fontsize=14,fontweight='bold')
ax.set_xticks([0,1],['LCC Evaluation','LCC Combined']);ax.set_ylabel('AUC');ax.set_ylim(0,1);polish(ax)
ax.legend(loc='upper left',ncols=3,fontsize=12)
footer(fig,'Full-scale uses CelebA Test-as-Val for selection / threshold calibration and a separate 12-epoch schedule.')
save(fig,'fig_fullscale_scaleup_summary',full_sources+[D18/'runs'/IDS18[2]/'config.json'],full_data,
     'MAST-PAD is strongest on LCC Evaluation; WBST is slightly higher on pooled LCC Combined in the secondary full-scale experiment.',
     'Uses exact per-run JSON, not rounded static values. Full-scale changes data size, schedule and selection protocol; no untouched source Test claim.')

missing = '''# Frequency counterfactual examples: source images needed\n\nNot generated: `fig_frequency_counterfactual_examples.png/.pdf`.\n\nThe repository contains run metrics, CSV manifests and bbox caches, but the 100K source JPEGs are not present in the inspected workspace. Existing EDA galleries and plots cannot establish the original crop geometry / sample membership and were not substituted for dataset images.\n\nTo complete this figure, mount the original CelebA-Spoof images (the run config uses a Kaggle-only `celeba_root`), choose one Real and one Attack entry from the frozen Train100K manifest, and resolve their valid bbox-cache records. Use `mini_crop_bgr`, `radial_masks` and `spectral_views` directly from notebook 17. Do not resample the training manifest or alter cached bboxes.\n\nFrozen rendering settings: crop1.5; 80×80 BGR float [0,1]; normalized radius sqrt(x²+y²)/sqrt(2×40²); centers [0.0833333333, 0.25, 0.4166666667]; sigma 0.10; gain 0.775; preserve DC; keep phase; clamp inverse-FFT output to [0,1]. Convert BGR to RGB only for display. Display clean log FFT magnitude as the fifth column. No subject names, IDs or paths in the figure.\n\nCaption: “Frequency views are used only during training; inference uses the clean spatial model.”\n'''
(OUT/'missing_frequency_examples_note.md').write_text(missing)
versions={'Python':platform.python_version(),'Matplotlib':matplotlib.__version__,'NumPy':np.__version__}
timestamp=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
payload={'generated_at':timestamp,'versions':versions,'figures':records,'missing':['fig_frequency_counterfactual_examples']}
(OUT/'figure_data.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
manifest=['# MAST-PAD figure material','',f'Generated: {timestamp}', '',
          'Six figures from existing artifacts, each with 300-DPI PNG and vector PDF; method flow also has SVG.',
          'No training, checkpoint editing, threshold fitting, or changes to existing experiment outputs.', '',
          '## Reproduce','', '`python3 slide_figures_mast_pad_100k/generate_figures.py`','',
          'Dependencies: '+', '.join(f'{k} {v}' for k,v in versions.items()),'',
          'Exact plotted values and complete curve arrays: [figure_data.json](figure_data.json).','',
          '## Interpretation','',
          '- Main experiment: MAST-PAD is the selected model; one training seed. No significance claim.',
          '- LCC Combined pools training, development and evaluation splits. LCC was not used to choose checkpoints or thresholds.',
          '- AUC is threshold-independent; the source-frozen threshold statement describes the overall evaluation protocol.',
          '- Contextual literature points have different protocols and come from the supplied prompt.',
          '- Full-scale is secondary and uses source Test-as-Val, with a separate training schedule.',
          '- Curves and metrics are read from individual run artifacts; stale top-level summary snapshots are not used.','']
for name,v in records.items():
    manifest += [f'## {name}', '', 'Files: '+', '.join(f'[{f}]({f})' for f in v['files']), '',
                 'Caption: '+v['caption'], '', 'Sources:', '']
    manifest += ['- `'+p+'`' for p in v['sources']]
    manifest += ['', 'Static fallback: **No**. Supplied contextual references: **'+('Yes' if v['supplied_context'] else 'No')+'**.', '',v['note'],'',
                 'Exact plotted values:', '', '```json',json.dumps(v['data'],indent=2,ensure_ascii=False),'```','']
manifest += ['## Not generated: frequency counterfactual examples','',
             'Original dataset images are not available locally. See [missing_frequency_examples_note.md](missing_frequency_examples_note.md).', '',
             '## Contact sheet','', '[preview_contact_sheet.png](preview_contact_sheet.png) is a navigation preview; use individual PNG/PDF/SVG files for slides and paper.','']
(OUT/'figure_manifest.md').write_text('\n'.join(manifest))
print(f'Generated {len(records)} figures in {OUT}')

# Compact visual index, regenerated with the figure set.
from PIL import Image, ImageDraw
files = sorted(OUT.glob('fig_*.png'))
sheet = Image.new('RGB', (1600, 1050), '#e5e9ee')
draw = ImageDraw.Draw(sheet)
for i, path in enumerate(files):
    im = Image.open(path); im.thumbnail((780,305))
    sheet.paste(im, ((i%2)*800+(800-im.width)//2, (i//2)*350+32))
    draw.text(((i%2)*800+15,(i//2)*350+8), path.stem, fill='black')
sheet.save(OUT/'preview_contact_sheet.png')
