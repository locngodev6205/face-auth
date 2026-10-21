"""Regenerate the measured all-data PAD report tables/figures from cached runs."""
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimize_pad_domain import ROOT,load_samples,reconstruct_crop,write_json,write_csv
from antispoof.lighting import LightingTransform,BrightnessThreshold


def main():
    base=ROOT/'output/pad_domain'
    outputs={key:base/name for key,name in dict(v1='2026-10-21_all80_v1',v2='2026-10-21_all80_v2',
                 robust='2026-10-21_all80_robust',adaptive='2026-10-21_all80_adaptive').items()}
    reports={k:json.loads((p/'report.json').read_text()) for k,p in outputs.items()}
    rows=[]
    for name,result in [('Current',reports['v1']['baseline']),('Threshold only',reports['v1']['threshold_only']),
                        ('Best clean fixed',reports['v2']['recommended']),('Robust fixed',reports['robust']['recommended'])]:
        rows.append(dict(method=name,FA=result['fake_as_real'],FR=result['real_as_fake'],
                         APCER=result['APCER'],BPCER=result['BPCER'],ACER=result['ACER']))
    adaptive=reports['adaptive']['recommended'];clean=reports['adaptive']['groups'][0]
    rows.append(dict(method='Brightness adaptive',FA=clean['fake_as_real'],FR=clean['real_as_fake'],
                     APCER=clean['APCER'],BPCER=clean['BPCER'],ACER=clean['ACER']))
    out=outputs['adaptive'];write_csv(out/'comparison.csv',rows)
    samples,_=load_samples(ROOT/'pad_captures')
    crops=[reconstruct_crop(s,adaptive['crop_factor']) for s in samples]
    transform=LightingTransform(json.loads(adaptive['profile']))
    adapter=BrightnessThreshold(dict(reference_luma=110.,scale=50.,slope=adaptive['slope'],max_offset=2.))
    times=[]
    for _ in range(5):
        for image in crops:
            start=time.perf_counter_ns();adapter(image,adaptive['threshold_logit']);transform(image)
            times.append((time.perf_counter_ns()-start)/1e6)
    timing=dict(host='Local Mac CPU; not Raspberry Pi',n_measurements=len(times),image_size=[80,80],
                lighting_and_adaptive_threshold_median_ms=float(np.median(times)),
                lighting_and_adaptive_threshold_p95_ms=float(np.percentile(times,95)))
    write_json(out/'transform_timing.json',timing)
    # Raw logits for baseline and effective d-t margins for the candidate.
    baseline={}
    with (outputs['v1']/'scores.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['candidate']=='crop1.5_00_none':baseline[row['file']]=float(row['score_logit'])-reports['v1']['baseline']['threshold_logit']
    with (out/'recommended_predictions.csv').open() as stream:
        predictions=list(csv.DictReader(stream))
    margins={r['file']:float(r['score_minus_threshold']) for r in predictions}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    x=np.arange(len(rows));axes[0].bar(x-.18,[r['APCER']*100 for r in rows],.36,label='Fake accepted')
    axes[0].bar(x+.18,[r['BPCER']*100 for r in rows],.36,label='Real rejected')
    axes[0].set_xticks(x,[r['method'] for r in rows],rotation=25,ha='right');axes[0].set_ylabel('Error (%)')
    axes[0].set_title('All 80 calibration images');axes[0].legend(fontsize=8)
    for ax,mapping,title in [(axes[1],baseline,'Current d - t'),(axes[2],margins,'Adaptive d - t(image)')]:
        for label,color in [('real','#13795b'),('fake','#c0392b')]:
            values=[mapping[s['file']] for s in samples if s['label']==label]
            ax.hist(values,bins=15,alpha=.6,color=color,label=label)
        ax.axvline(0,color='black',linestyle='--');ax.set_title(title);ax.set_xlabel('Decision margin');ax.legend()
    fig.suptitle('Exploratory fit on all labelled images — no held-out evaluation',fontsize=11)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    session_rows=[]
    for session in sorted({s['session'] for s in samples}):
        selected=[r for r in predictions if Path(r['file']).parent.name==session]
        session_rows.append(dict(session=session,n_real=sum(r['label']=='real' for r in selected),
            n_fake=sum(r['label']=='fake' for r in selected),
            fake_as_real=sum(r['label']=='fake' and r['predicted']=='real' for r in selected),
            real_as_fake=sum(r['label']=='real' and r['predicted']=='fake' for r in selected)))
    write_csv(out/'session_results.csv',session_rows)
    # AUC computed directly from all cross-class pairs, ties receive 0.5.
    auc={}
    for name,score_map in [('baseline',baseline),('candidate_centered_margin',margins)]:
        real=np.array([score_map[s['file']] for s in samples if s['label']=='real'])
        fake=np.array([score_map[s['file']] for s in samples if s['label']=='fake'])
        delta=real[:,None]-fake[None,:]
        auc[name]=float(np.mean((delta>0)+.5*(delta==0)))
    write_json(out/'diagnostics.json',dict(AUC=auc,session_results=session_rows,timing=timing))
    print(json.dumps(dict(comparison=rows,diagnostics=auc,timing=timing),indent=2))


if __name__=='__main__':main()
