"""Reproduce full-scale presentation figures from existing artifacts; no model runs."""
from pathlib import Path
import os,json,csv,platform,ast
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('MPLCONFIGDIR','/tmp/mast_fullscale_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
BASE=ROOT/'antispoof/notebooks/final/binary_crossdomain_v2'
FULL=BASE/'output/18_fullscale_C_P3SF_R7SC_crop15'
CASIA=BASE/'output/casia_train_test_C_P3SF_R7SC_100k_fullscale'
D17=BASE/'output/17_scale100k_C_P3SF_R7SC_crop15'
PROMPT=ROOT/'md/mast_pad_codex_figure_prompts_FULLSCALE.md'
LABELS=['Clean','WBST','MAST-PAD']
IDS=['C_FULL_crop15','P3_SF_FULL_crop15','R7_SC_FULL_crop15']
COLORS=['#7c8995','#397ab7','#008b83']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':15,'axes.titlesize':18,
 'xtick.labelsize':13,'ytick.labelsize':13,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
records={}
def read(p):return json.loads(p.read_text())
def rows(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def rel(p):return str(p.relative_to(ROOT))
def save(fig,name,sources,data,caption,caveat='',refs=None,svg=False,static=False,role='main'):
 files=[]
 for ext in ['png','pdf']+(['svg'] if svg else []):
  f=OUT/f'{name}.{ext}';fig.savefig(f,dpi=300,bbox_inches='tight',pad_inches=.2);files.append(f.name)
 plt.close(fig)
 records[name]={'files':files,'sources':[rel(p) for p in sources],'data':data,'caption':caption,
  'caveats':caveat,'reference_keys':refs or [],'static_fallback_used':False,
  'supplied_literature_values_used':static,'role':role}
def polish(ax):ax.set_axisbelow(True);ax.grid(axis='y',color='#e6ecef',lw=.8)
def foot(fig,s):fig.text(.5,.025,s,ha='center',va='bottom',fontsize=11,color='#526773')
LCC_SOURCES=[FULL/'runs'/r/'metrics/lcc_combined_metrics.json' for r in IDS]
CASIA_SOURCES=[CASIA/'runs'/r/'combined/video_metrics.json' for r in IDS]
lcc=[read(p)['auc'] for p in LCC_SOURCES];casia=[read(p)['auc'] for p in CASIA_SOURCES]
complexity=FULL/'complexity_report.json';params=read(complexity)['shared_clean_inference']['params_m']
best=[read(FULL/'runs'/r/'completion.json')['best_epoch'] for r in IDS]
# Reference point values are exactly those requested; verification records distinguish
# original tables from baseline tables, and average vs selected-best reporting.
references={
 'Graph Transformer':{'urls':['https://www.esann.org/sites/default/files/proceedings/2023/ES2023-14.pdf'],
  'verification':'AUC 0.833 verified in Table 1; 10.84M verified in architecture section. Abstract/result prose differ; use the requested table entry.'},
 'AENet':{'urls':['https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1485_ECCV_2020_paper.php','https://github.com/kprokofi/light-weight-face-anti-spoofing'],
  'verification':'LCC AUC 0.868 and 11.22M verified in supplied repository benchmark; paper URL identifies the model.'},
 'MADDG':{'urls':['https://openaccess.thecvf.com/content_CVPR_2019/html/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.html','https://arxiv.org/pdf/2207.09868','https://arxiv.org/html/2203.14325v1'],
  'verification':'AUC 84.51% cross-checked in AMEL and PatchNet baseline tables, O&M&I to C. Original CVF page could not be fetched in this session.'},
 'AMEL':{'urls':['https://arxiv.org/abs/2207.09868','https://arxiv.org/pdf/2207.09868'],
  'verification':'AUC 94.39% verified in AMEL O&M&I to C table.'},
 'PatchNet':{'urls':['https://openaccess.thecvf.com/content/CVPR2022/html/Wang_PatchNet_A_Simple_Face_Anti-Spoofing_Framework_via_Fine-Grained_Patch_Recognition_CVPR_2022_paper.html','https://arxiv.org/html/2203.14325v1'],
  'verification':'AUC 94.58% verified in original arXiv paper Table 8; CVF page could not be fetched.'},
 'GAC-FAS':{'urls':['https://openaccess.thecvf.com/content/CVPR2024/html/Le_Gradient_Alignment_for_Cross-Domain_Face_Anti-Spoofing_CVPR_2024_paper.html','https://arxiv.org/html/2402.18817v1'],
  'verification':'AUC 95.16% verified in Table 1, reported with 0.09 standard deviation. Requested point estimate only is plotted; this is not a matched seed comparison.'},
 'SA-FAS':{'urls':['https://openaccess.thecvf.com/content/CVPR2023/html/Sun_Rethinking_Domain_Generalization_for_Face_Anti-Spoofing_Separability_and_Alignment_CVPR_2023_paper.html','https://arxiv.org/pdf/2303.13662'],
  'verification':'AUC 95.37% verified in original arXiv paper Table 1. This selected-best reporting differs from its Table 2 mean/std protocol. CVF page could not be fetched.'}}

fig,ax=plt.subplots(figsize=(14,7.5));fig.subplots_adjust(top=.73,bottom=.21)
fig.suptitle('Main results: full-scale cross-domain AUC',fontsize=23,y=.985)
fig.text(.5,.915,'Full-scale CelebA source training; frozen cross-domain evaluation',ha='center',fontsize=15)
for i,(label,color) in enumerate(zip(LABELS,COLORS)):
 xs=np.arange(2)+(i-1)*.24;ys=[lcc[i],casia[i]]
 ax.bar(xs,ys,width=.23,color=color,label=label)
 for x,y in zip(xs,ys):ax.text(x,y+.012,f'{y:.3f}',ha='center',fontweight='bold',fontsize=16)
ax.set_xticks([0,1],['LCC Combined\n(image-level)','CASIA Combined\n(reconstructed video-level)'])
ax.set_ylim(0,1.09);ax.set_yticks(np.arange(0,1.01,.2));ax.set_ylabel('AUC');polish(ax)
ax.legend(loc='upper left',ncols=3,fontsize=13)
fig.text(.275,.825,'WBST and MAST-PAD nearly tied',ha='center',fontsize=14,color=COLORS[1])
fig.text(.735,.825,f'MAST-PAD +{100*(casia[2]-casia[0]):.2f} pp vs Clean',ha='center',fontsize=14,color=COLORS[2],fontweight='bold')
foot(fig,'CASIA: 358/360 scorable reconstructed videos; combined deduplicates overlapping train/test frame IDs.')
save(fig,'fig_main_ablation_two_target_auc',LCC_SOURCES+CASIA_SOURCES+[CASIA/'evaluation_protocol.json'],
 {'LCC Combined':dict(zip(LABELS,lcc)),'CASIA Combined':dict(zip(LABELS,casia)),'CASIA_gain_pp':100*(casia[2]-casia[0])},
 'Full-scale WBST and MAST-PAD are nearly tied on LCC Combined; MAST-PAD improves CASIA Combined reconstructed-video AUC by 1.48 percentage points over Clean.',
 'LCC is image-level, CASIA is reconstructed video-level (mean score; ≥4 valid frames), so there is no cross-target pooled AUC. Source checkpoint/threshold selection uses CelebA Test-as-Val. Filename follows the prompt; this figure is labeled Main, not ablation.')

context={'Graph Transformer':{'params_M':10.84,'auc':.833},'AENet':{'params_M':11.22,'auc':.868},'MAST-PAD':{'params_M':params,'auc':lcc[2]}}
fig,ax=plt.subplots(figsize=(12,7));fig.subplots_adjust(bottom=.19,top=.86,right=.88)
ax.set_title('LCC: deployment size and contextual AUC',pad=18)
for name,v in context.items():ax.scatter(v['params_M'],v['auc'],s=240 if name=='MAST-PAD' else 110,color=COLORS[2] if name=='MAST-PAD' else '#8296b1',zorder=3)
ax.annotate('MAST-PAD\n0.434M params, AUC 0.860',(params,lcc[2]),xytext=(17,17),textcoords='offset points',color=COLORS[2],fontweight='bold',fontsize=15)
ax.annotate('AENet',(11.22,.868),xytext=(-12,16),ha='right',textcoords='offset points',fontsize=15)
ax.annotate('Graph Transformer',(10.84,.833),xytext=(-15,-25),ha='right',textcoords='offset points',fontsize=15)
ax.set_xscale('log');ax.set_xlim(.25,22);ax.set_ylim(.82,.885);ax.set_xticks([.3,.5,1,2,5,10,20]);ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('Parameters (M), log scale');ax.set_ylabel('LCC AUC');polish(ax)
foot(fig,'Contextual references; protocols differ.')
save(fig,'fig_lcc_auc_vs_params_context',[LCC_SOURCES[2],complexity,PROMPT],context,
 'MAST-PAD achieves pooled LCC AUC 0.860 with 434,434 clean deployment parameters; literature points provide context.',
 'Different training/evaluation/preprocessing protocols and parameter-count conventions; no controlled ranking claim.',refs=['Graph Transformer','AENet'],static=True)

lit={'MADDG':.8451,'MAST-PAD':casia[2],'AMEL':.9439,'PatchNet':.9458,'GAC-FAS':.9516,'SA-FAS':.9537}
fig,ax=plt.subplots(figsize=(14,8));fig.subplots_adjust(left=.35,bottom=.23,top=.87,right=.92)
ax.set_title('CASIA AUC: literature context',pad=20)
for i,(name,y) in enumerate(lit.items()):
 c=COLORS[2] if name=='MAST-PAD' else '#8498b4'
 ax.scatter(y,i,s=240 if name=='MAST-PAD' else 110,color=c,zorder=3)
 ax.text(y+.004,i,f'{y:.4f}',va='center',fontsize=15,fontweight='bold' if name=='MAST-PAD' else 'normal',color=c)
ax.set_yticks(range(len(lit)),[n+'\n'+('CelebA-full → reconstructed CASIA Combined' if n=='MAST-PAD' else 'O&M&I → C') for n in lit])
ax.invert_yaxis();ax.set_xlim(.82,1);ax.set_xlabel('AUC');ax.grid(axis='x',color='#e6ecef');ax.set_axisbelow(True)
for tick,name in zip(ax.get_yticklabels(),lit):
 if name=='MAST-PAD':tick.set_color(COLORS[2]);tick.set_fontweight('bold')
fig.text(.5,.1,'Different source-domain protocols; contextual comparison only.',ha='center',fontsize=13)
foot(fig,'MAST-PAD deployment: 0.434M params. CASIA point: deduplicated Kaggle image copy, reconstructed video scores.')
save(fig,'fig_casia_auc_literature_context',[CASIA_SOURCES[2],CASIA/'evaluation_protocol.json',PROMPT,complexity],lit,
 'Contextual CASIA AUC points, separating O&M&I → C literature protocols from CelebA-full → reconstructed CASIA Combined.',
 'MAST-PAD uses a nonofficial extracted-image combined copy, 358 videos and one trained checkpoint. Literature numbers include selected-best results and a mean (GAC-FAS); no matched benchmark or significance claim.',refs=['MADDG','AMEL','PatchNet','GAC-FAS','SA-FAS'],static=True)

fig,ax=plt.subplots(figsize=(16,9));ax.set_xlim(0,16);ax.set_ylim(0,9);ax.axis('off')
ax.text(.5,8.55,'MAST-PAD: margin-aware spectral supervision',fontsize=24,fontweight='bold',color='#173646')
ax.text(.5,8,'Training: select the view with the lowest label-aligned margin',fontsize=16,color='#526773')
def box(x,y,w,h,s,size=15,color='#eaf5f4'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12,rounding_size=.15',facecolor=color,edgecolor='#38877f',lw=1.4));ax.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,color='#173646')
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color='#38877f',lw=1.6))
box(.5,4.35,1.6,1.15,'Clean\ncrop')
for y,s in [(6.2,'LOW'),(4.55,'MID'),(2.9,'HIGH')]:
 box(2.9,y,2.15,1.1,s+'\nattenuation',14);arrow((2.2,4.92),(2.78,y+.55));arrow((5.17,y+.55),(6.05,4.92))
box(6.2,4.3,2.6,1.25,'Shared\nMiniFASNetV2 PAD',15,'#eaf1fb')
arrow((1.3,5.62),(1.3,7.55));arrow((1.3,7.55),(7.5,7.55));arrow((7.5,7.55),(7.5,5.7));ax.text(4.3,7.72,'Clean view: same model',ha='center',fontsize=12,color='#526773')
box(9.55,4.3,3.1,1.25,'Label-aligned margins\n'+r'$m_{low},\ m_{mid},\ m_{high}$',14);arrow((8.95,4.92),(9.4,4.92))
box(13.35,4.3,2.05,1.25,'Worst view\n'+r'$k^*=\arg\min_k m_k$',14);arrow((12.8,4.92),(13.2,4.92))
box(10.45,2.05,4.9,1.15,'Harmful only if\n'+r'$m_{worst}<m_{clean}$',17);arrow((14.38,4.16),(14.38,3.35));arrow((8.05,4.16),(10.3,2.8));ax.text(8.6,3.3,r'$m_{clean}$',fontsize=15)
box(.5,.65,8.2,1.0,r'$L=0.75\,CE_{clean}+0.25\,CE_{harmful}+L_{aux}$',19);arrow((12.85,1.91),(8.84,1.18))
ax.text(.65,2.4,'Spectral views are training-only.',fontsize=16,color=COLORS[2],fontweight='bold')
ax.text(.65,1.98,'Warmup: clean only. No harmful sample: spectral loss is zero.',fontsize=12,color='#526773')
ax.text(.5,.03,'Inference: crop1.5 → 80×80 → MiniFASNetV2 → Real / Attack',fontsize=16,color='#173646')
save(fig,'fig_mast_pad_method_flow',[BASE/'18_fullscale_C_P3SF_R7SC_crop15.ipynb',FULL/'runs'/IDS[2]/'config.json'],
 {'active_loss_weights':[.75,.25],'warmup_epochs':1,'crop_factor':1.5,'input_size':[80,80],'selector':'argmin label-aligned margin','gate':'worst margin < clean margin'},
 'Select the minimum-margin spectral counterfactual and supervise it only when it degrades the clean margin; deployment is the spatial binary PAD model.',
 'Selection: eval/no_grad, BN-safe; harmful CE averages harmful samples; clean auxiliary losses are training-only.',svg=True)

sp=FULL/'runs'/IDS[2]/'spectral_diagnostics.csv';rr=rows(sp);epochs=[int(r['epoch']) for r in rr];last=rr[-1]
curves={k:[float(r[k]) for r in rr] for k in ['harmful_fraction','margin_drop','spectral_flip_rate']}
bands={k.upper():float(last[k+'_fraction']) for k in ['low','mid','high']}
fig,(ax,ab)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1.7,1]});fig.subplots_adjust(top=.82,bottom=.23,wspace=.46)
fig.suptitle('Full-scale MAST-PAD: training diagnostics',fontsize=22)
ax.plot(epochs,curves['harmful_fraction'],color=COLORS[2],lw=2.5,label='Harmful fraction');ax.plot(epochs,curves['spectral_flip_rate'],color='#9867a7',lw=2.5,label='Prediction flip rate')
ax.set_ylim(0,1);ax.set_ylabel('Sample fraction');ax.set_xlabel('Epoch');ax.set_title('A  Active training',loc='left');polish(ax)
twin=ax.twinx();twin.spines['right'].set_visible(True);twin.plot(epochs,curves['margin_drop'],color=COLORS[1],lw=2.5,ls='--',label='Mean margin drop');twin.set_ylim(0,.7);twin.set_ylabel('Mean margin drop (logit units)',color=COLORS[1])
ax.axvline(best[2],color='#536674',ls='--');ax.axvline(epochs[-1],color='#536674',ls=':');ax.text(best[2]+.13,.94,f'Best: {best[2]}',fontsize=11);ax.text(epochs[-1]-.13,.89,f'Stop: {epochs[-1]}',ha='right',fontsize=11)
ax.set_xticks([2,4,6,7,8,10,11]);lines=ax.lines[:2]+twin.lines;ax.legend(lines,[l.get_label() for l in lines],loc='lower left',fontsize=11)
ax.annotate(f'Final flip rate: {curves["spectral_flip_rate"][-1]*100:.3f}%',(epochs[-1],curves['spectral_flip_rate'][-1]),xytext=(-5,21),textcoords='offset points',ha='right',fontsize=11,color='#9867a7')
ab.bar(list(bands),np.array(list(bands.values()))*100,color=['#9bc6df',COLORS[1],COLORS[2]],width=.65);ab.set_ylim(0,52);ab.set_ylabel('Selected band (%)');ab.set_title(f'B  Final epoch {epochs[-1]}',loc='left');polish(ab)
for i,v in enumerate(bands.values()):ab.text(i,v*100+1,f'{v*100:.2f}%',ha='center',fontsize=15,fontweight='bold')
fig.text(.5,.105,'Margin degradation is common while outright spectral prediction flips remain rare.',ha='center',fontsize=13)
foot(fig,'Active epochs only; epoch 1 is clean warmup. Left and right axes have different units.')
save(fig,'fig_mast_pad_training_diagnostics',[sp,FULL/'runs'/IDS[2]/'completion.json'],
 {'epochs':epochs,'curves':curves,'best_epoch':best[2],'stop_epoch':epochs[-1],'final_band_fractions':bands},
 'Margin degradation is common while outright spectral prediction flips remain rare.',
 'No static fallback. Band selection counts all samples. Flip rate is clean-correct / selected-view-wrong samples divided by total, not all prediction changes.')

history_sources=[FULL/'runs'/r/'training_history.csv' for r in IDS];hist=[rows(p) for p in history_sources]
fig,axes=plt.subplots(1,2,figsize=(15,7));fig.subplots_adjust(top=.82,bottom=.25,wspace=.26)
fig.suptitle('Full-scale: source-selection training curves',fontsize=22)
curve_data={}
for label,color,h,b in zip(LABELS,COLORS,hist,best):
 e=[int(r['epoch']) for r in h];auc=[float(r['val_auc']) for r in h];acer=[100*float(r['val_acer']) for r in h]
 curve_data[label]={'epochs':e,'source_selection_auc':auc,'source_selection_acer_percent':acer,'best_epoch':b}
 for ax,v in zip(axes,[auc,acer]):ax.plot(e,v,color=color,lw=2,marker='o',ms=4,label=label);ax.scatter(b,v[e.index(b)],marker='*',s=210,color=color,edgecolors='#173646',lw=.6,zorder=4)
for ax,title,ylabel in zip(axes,['A  Source-selection AUC','B  Source-selection ACER'],['AUC','ACER (%)']):
 ax.set_title(title,loc='left');ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.set_xticks(range(1,13));polish(ax);ax.axvspan(.7,1.3,color='#edf1f4',zorder=0)
 for m in [4,8]:ax.axvline(m,color='#bac5cc',ls=':',lw=1)
axes[0].legend(loc='lower right',fontsize=12)
fig.text(.5,.115,'★ Selected checkpoints: Clean 9 · WBST 6 · MAST-PAD 7',ha='center',fontsize=14)
foot(fig,'Selection set: CelebA Test-as-Val, not untouched Test. Warmup: epoch 1 for spectral methods; LR milestones: 4 and 8.')
save(fig,'fig_training_curves_fullscale',history_sources+[FULL/'runs'/r/'completion.json' for r in IDS],curve_data,
 'Full-scale source-selection curves, with stars marking the original chosen checkpoints.',
 'Source selection and threshold calibration reuse official CelebA Test-as-Val. ACER here is a training diagnostic, not a headline target comparison; axes span observed values.')

support_sources=[D17/'runs'/r/'metrics/lcc_combined_metrics.json' for r in ['C_100K_crop15','P3_SF_100K_crop15','R7_SC_100K_crop15']]
support=[read(p)['auc'] for p in support_sources]
fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(top=.83,bottom=.19)
ax.set_title('Supporting matched100K development evidence',pad=18);ax.bar(LABELS,support,color=COLORS,width=.55);ax.set_ylim(0,1);ax.set_ylabel('LCC Combined AUC');polish(ax)
for i,v in enumerate(support):ax.text(i,v+.015,f'{v:.3f}',ha='center',fontsize=17,fontweight='bold')
foot(fig,'Supporting / appendix evidence; the main figures use the full-scale experiment.')
save(fig,'fig_supporting_100k_lcc_auc',support_sources,dict(zip(LABELS,support)),
 'Supporting matched100K development results, reserved for backup slides or appendix.',role='supporting')

(OUT/'missing_frequency_examples_note.md').write_text('''# Frequency examples: missing source crops\n\nFigure 7 is not generated. No eligible Real + Attack source crops from CelebA-full are available in the local artifacts inspected. Personal gallery and LFW demo images are not substituted for labeled source PAD samples; CASIA prediction CSVs are not source crops.\n\nSupply one frozen CelebA training Real and one Attack image with valid SCRFD bbox records (or their exact 1.5×/80×80 crops). Use notebook18 `mini_crop_bgr`, `radial_masks` and `spectral_views`, centers [0.0833333333,0.25,0.4166666667], sigma 0.10, gain 0.775, DC preserved. Display BGR input as RGB only for rendering; log FFT magnitude from the clean crop. No identities or filenames on the figure.\n\nCaption: “Frequency counterfactuals are training-only; deployment remains spatial-only.”\n''')
timestamp=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
versions={'Python':platform.python_version(),'Matplotlib':matplotlib.__version__,'NumPy':np.__version__}
data={'generated_at':timestamp,'versions':versions,'method_to_run':dict(zip(LABELS,IDS)),
 'figures':records,'references':references,'missing':['fig_frequency_counterfactual_examples']}
(OUT/'figure_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
manifest=['# MAST-PAD full-scale figure manifest','',f'Generated: {timestamp}', '',
 'Seven figures; PNG 300 DPI and vector PDF, plus SVG for the method diagram. Figure 7 is missing source images; supporting 100K is explicitly appendix-only.', '',
 '## Reproduction','', '`python3 slide_figures_mast_pad_fullscale/generate_figures.py`','',
 'Packages: '+', '.join(f'{k} {v}' for k,v in versions.items()), '',
 'No model training, inference, recalibration, checkpoint editing or changes to existing experiment outputs.', '',
 '## Run mapping','',*['- '+label+': `'+run+'`' for label,run in zip(LABELS,IDS)],'',
 '## Common protocol caveats','',
 '- Full-scale selection/calibration uses source CelebA Test-as-Val; it is not untouched source Test.',
 '- LCC Combined pools three image splits; CASIA Combined is a unique union of reconstructed frame IDs, with test-copy precedence.',
 '- CASIA AUC in headline figures is video-level (358 scored / 360 candidates; minimum four valid frames). LCC AUC is image-level. Do not average the two target AUCs.',
 '- CASIA copy is not verified to be the official subject-disjoint benchmark; its train/test copies overlap.',
 '- Reference source domains, preprocessing, evaluation units and result selection/statistics differ. Context only; no controlled SOTA or significance claim.',
 '- Existing CASIA results are already present; this script does not run target evaluation.', '',
 'Exact plotted data and curve arrays: [figure_data.json](figure_data.json).','']
for name,v in records.items():
 manifest += [f'## {name}','', 'Role: '+v['role'], '', 'Files: '+', '.join(f'[{f}]({f})' for f in v['files']), '',
  'Caption: '+v['caption'],'', 'Sources:','',*['- `'+p+'`' for p in v['sources']], '',
  'Static fallback used: **No**. Supplied literature values used: **'+('Yes' if v['supplied_literature_values_used'] else 'No')+'**.','',
  'Protocol caveats: '+v['caveats'],'','Reference keys: '+(', '.join(v['reference_keys']) or 'none'),'',
  'Exact numeric values:','','```json',json.dumps(v['data'],indent=2,ensure_ascii=False),'```','']
manifest += ['## External references and verification','']
for name,r in references.items():
 manifest += ['### '+name,'',r['verification'],'',*[f'- [Source {i+1}]({u})' for i,u in enumerate(r['urls'])],'']
manifest += ['## Missing figure','', '[Frequency counterfactual examples — source data needed](missing_frequency_examples_note.md)','']
(OUT/'figure_manifest.md').write_text('\n'.join(manifest))
files=sorted(OUT.glob('fig_*.png'));sheet=Image.new('RGB',(1600,1400),'#e5e9ee');draw=ImageDraw.Draw(sheet)
for i,p in enumerate(files):
 im=Image.open(p);im.thumbnail((780,305));sheet.paste(im,((i%2)*800+(800-im.width)//2,(i//2)*350+32));draw.text(((i%2)*800+12,(i//2)*350+7),p.stem,fill='black')
sheet.save(OUT/'preview_contact_sheet.png')
print(f'Generated {len(records)} figure sets in {OUT}')
