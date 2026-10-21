"""Explore bounded brightness-conditioned PAD thresholds on cached all-data scores."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import cv2
from optimize_pad_domain import (ROOT,load_samples,reconstruct_crop,perturb,write_json,write_csv,probability,sha)
from refine_pad_robustness import robust_fit
import config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    samples,snapshots=load_samples(ROOT/'pad_captures')
    y=np.array([s['label']=='real' for s in samples],int)
    changes=('original','gain_0.9','gain_1.1','gamma_0.95','gamma_1.05','jpeg90','blur')
    file_index={s['file']:i for i,s in enumerate(samples)}
    matrices={}
    with (args.run/'scores.csv').open() as stream:
        for row in csv.DictReader(stream):
            matrix=matrices.setdefault(row['candidate'],np.full((7,len(samples)),np.nan))
            matrix[changes.index(row['perturbation']),file_index[row['file']]]=float(row['score_logit'])
    configs={}
    source_runs=json.loads((args.run/'report.json').read_text())['source_runs']
    for source_run in source_runs:
        run=Path(source_run)
        with (run/'candidates.csv').open() as stream:
            for row in csv.DictReader(stream):
                configs[f'{run.name}:{row["candidate"]}']=row
    luminances={};rows=[];cache={}
    for name,scores in matrices.items():
        assert np.isfinite(scores).all()
        factor=float(configs[name]['crop_factor'])
        if factor not in luminances:
            crops=[reconstruct_crop(s,factor) for s in samples]
            luminances[factor]=np.array([[float(cv2.cvtColor(perturb(c,change),cv2.COLOR_BGR2YCrCb)[:,:,0].mean())
                                         for c in crops] for change in changes])
        for slope in (-3,-2,-1.5,-1,-.5,0,.5,1,1.5,2,3):
            offset=np.clip(slope*(luminances[factor]-110)/50,-2,2)
            centered=scores-offset
            fit=robust_fit(centered,y)
            if fit is None:
                continue
            objective,t,ap,bp,ac=fit
            row=dict(candidate=name,slope=slope,crop_factor=factor,profile=configs[name]['profile'],
                     threshold_logit=t,clean_ACER=float(ac[0]),clean_APCER=float(ap[0]),clean_BPCER=float(bp[0]),
                     augmented_mean_ACER=float(ac.mean()),augmented_worst_ACER=float(ac.max()),
                     augmented_mean_APCER=float(ap.mean()),augmented_mean_BPCER=float(bp.mean()))
            rows.append(row);cache[(name,slope)]=(centered,offset)
        print(f'Adaptive fit {name}',flush=True)
    write_csv(args.output/'candidates.csv',rows)
    objective=lambda r:(r['augmented_mean_ACER'],r['augmented_worst_ACER'],r['augmented_mean_APCER'],r['clean_ACER'],abs(r['slope']))
    best=min(rows,key=objective);fixed=min((r for r in rows if r['slope']==0),key=objective)
    centered,offset=cache[(best['candidate'],best['slope'])]
    from calibrate_domain_threshold import metrics
    group_rows=[dict(metrics(values,y,best['threshold_logit']),perturbation=change)
                for change,values in zip(changes,centered)]
    source=json.loads(Path(config.PAD_RUNTIME_CONFIG_PATH).read_text())
    source.update(model_file=config.PAD_MODEL_PATH,bbox_expansion_factor=best['crop_factor'],
                  lighting=json.loads(best['profile']),apply_gamma=False,
                  calibrated_logit_threshold=best['threshold_logit'],
                  predictor_threshold_probability=probability(best['threshold_logit']),
                  adaptive_threshold=dict(reference_luma=110.,scale=50.,slope=best['slope'],max_offset=2.),
                  threshold_source='All local labelled captures plus synthetic perturbations; bounded brightness threshold; no held-out',
                  threshold_calibration=dict(report=str((args.output/'report.json').resolve()),heldout=False))
    write_json(args.output/'candidate_runtime.json',source)
    write_csv(args.output/'recommended_predictions.csv',[
        dict(file=s['file'],label=s['label'],score_logit=float(d+o),effective_threshold=float(best['threshold_logit']+o),
             score_minus_threshold=float(d-best['threshold_logit']),predicted='real' if d>=best['threshold_logit'] else 'fake')
        for s,d,o in zip(samples,centered[0],offset[0])])
    write_csv(args.output/'recommended_robustness.csv',group_rows)
    report=dict(run_class='exploratory_all_data_adaptive_threshold_fit',recommended=best,fixed_control=fixed,
                groups=group_rows,n_parameter_combinations=len(matrices)*11,heldout=False,
                command=sys.argv,source_sha256=sha(Path(__file__)),model_sha256=sha(Path(config.PAD_MODEL_PATH)))
    write_json(args.output/'report.json',report)
    (args.output/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    for path,digest in snapshots.items():
        assert sha(Path(path))==digest
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
