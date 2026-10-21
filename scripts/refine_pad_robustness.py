"""Fit a lightweight camera candidate to all labelled crops plus synthetic perturbations."""
import argparse
import csv
import json
from pathlib import Path
import time
import sys
import numpy as np
import onnxruntime as ort
import cv2
from optimize_pad_domain import (ROOT,load_samples,reconstruct_crop,perturb,LightingTransform,
                                 preprocess_batch,write_csv,write_json,sha,probability,metrics)
import config


def robust_fit(scores, labels, clean_limit=.16):
    values=np.unique(scores)
    thresholds=np.r_[np.nextafter(values[0],-np.inf),(values[:-1]+values[1:])/2,np.nextafter(values[-1],np.inf)]
    real=labels==1
    best=None
    for t in thresholds:
        pred=scores>=t
        ap=(pred[:,~real]).mean(axis=1)
        bp=(~pred[:,real]).mean(axis=1)
        ac=(ap+bp)/2
        if ac[0]>clean_limit:
            continue
        objective=(float(ac.mean()),float(ac.max()),float(ap.mean()),float(ac[0]))
        if best is None or objective<best[0]:
            best=(objective,float(t),ap,bp,ac)
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    samples,snapshots=load_samples(ROOT/'pad_captures')
    y=np.array([s['label']=='real' for s in samples],int)
    first_manifest=[(s['file'],s['label'],s['sha256']) for s in samples]
    candidates=[]
    for run in args.runs:
        with (run/'manifest.csv').open() as stream:
            previous=[(r['file'],r['label'],r['sha256']) for r in csv.DictReader(stream)]
        assert first_manifest==previous,'Manifest changed between stages'
        with (run/'candidates.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        rank=sorted(rows,key=lambda r:(float(r['ACER']),float(r['APCER'])))
        shortlist={r['candidate'] for r in rank[:12]}
        for method in {json.loads(r['profile'])['method'] for r in rows}:
            shortlist.add(next(r['candidate'] for r in rank if json.loads(r['profile'])['method']==method))
        shortlist.update(r['candidate'] for r in rows if json.loads(r['profile'])['method']=='none')
        candidates.extend(dict(r,source_run=str(run),id=f'{run.name}:{r["candidate"]}')
                          for r in rows if r['candidate'] in shortlist)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(config.PAD_MODEL_PATH,sess_options=options,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name
    def score(images):
        outputs=[]
        for start in range(0,len(images),16):
            batch=preprocess_batch(images[start:start+16],80,apply_gamma=False,convert_rgb=False)
            logits=session.run(None,{input_name:batch})[0]
            outputs.extend((logits[:,0]-logits[:,1]).astype(float))
        return np.array(outputs)
    changes=('original','gain_0.9','gain_1.1','gamma_0.95','gamma_1.05','jpeg90','blur')
    crop_cache={};grid=[];score_rows=[];stored={}
    for candidate in candidates:
        factor=float(candidate['crop_factor']);profile=json.loads(candidate['profile'])
        if factor not in crop_cache:
            crop_cache[factor]=[reconstruct_crop(s,factor) for s in samples]
        transform=LightingTransform(profile)
        scores=np.stack([score([transform(perturb(c,change)) for c in crop_cache[factor]]) for change in changes])
        assert np.isfinite(scores).all()
        stored[candidate['id']]=scores
        fit=robust_fit(scores,y)
        for change,vals in zip(changes,scores):
            score_rows.extend(dict(candidate=candidate['id'],perturbation=change,file=s['file'],
                                   label=s['label'],score_logit=float(value)) for s,value in zip(samples,vals))
        if fit is not None:
            objective,t,ap,bp,ac=fit
            m=metrics(scores[0],y,t)
            m.update(candidate=candidate['id'],crop_factor=factor,profile=json.dumps(profile,sort_keys=True),
                     augmented_mean_ACER=float(ac.mean()),augmented_worst_ACER=float(ac.max()),
                     augmented_mean_APCER=float(ap.mean()),augmented_mean_BPCER=float(bp.mean()))
            grid.append(m)
        print(f'Robust fit {candidate["id"]}: '+ ('no feasible threshold' if fit is None else str(objective)),flush=True)
    write_csv(args.output/'scores.csv',score_rows)
    write_csv(args.output/'candidates.csv',grid)
    best=min(grid,key=lambda r:(r['augmented_mean_ACER'],r['augmented_worst_ACER'],r['augmented_mean_APCER'],r['ACER']))
    group_rows=[dict(metrics(values,y,best['threshold_logit']),perturbation=change)
                for change,values in zip(changes,stored[best['candidate']])]
    write_csv(args.output/'recommended_robustness.csv',group_rows)
    source=json.loads(Path(config.PAD_RUNTIME_CONFIG_PATH).read_text())
    source.update(model_file=config.PAD_MODEL_PATH,bbox_expansion_factor=best['crop_factor'],
                  lighting=json.loads(best['profile']),apply_gamma=False,
                  calibrated_logit_threshold=best['threshold_logit'],
                  predictor_threshold_probability=probability(best['threshold_logit']),
                  threshold_source='All local labelled captures plus synthetic perturbations; no held-out',
                  threshold_calibration=dict(report=str((args.output/'report.json').resolve()),heldout=False))
    write_json(args.output/'candidate_runtime.json',source)
    report=dict(run_class='exploratory_all_data_robustness_fit',recommended=best,groups=group_rows,
                n_candidates=len(candidates),clean_ACER_limit=.16,heldout=False,
                source_runs=[str(p.resolve()) for p in args.runs],model_sha256=sha(Path(config.PAD_MODEL_PATH)),
                command=sys.argv,source_sha256=sha(Path(__file__)),lighting_sha256=sha(ROOT/'antispoof/lighting.py'))
    write_json(args.output/'report.json',report)
    (args.output/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'lighting_source.py').write_bytes((ROOT/'antispoof/lighting.py').read_bytes())
    for path,digest in snapshots.items():
        assert sha(Path(path))==digest
    for s in samples:
        assert sha(Path(s['file']))==s['sha256']
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
