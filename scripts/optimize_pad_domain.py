"""Exploratory all-data PAD lighting/crop/threshold fitting; no held-out claims.

python3 face_auth/scripts/optimize_pad_domain.py --output face_auth/output/pad_domain/run01
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from antispoof.lighting import LightingTransform
from antispoof.preprocess import preprocess_batch, crop_for_runtime
from capture_domain_faces import expanded_bounds
from calibrate_domain_threshold import metrics, select_threshold, probability


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_samples(root):
    samples, snapshots, known = [], {}, {}
    for labels_path in sorted(root.glob('*/labels.json')):
        meta_path = labels_path.with_name('metadata.json')
        labels, metadata = json.loads(labels_path.read_text()), json.loads(meta_path.read_text())
        snapshots[str(labels_path)] = sha(labels_path)
        snapshots[str(meta_path)] = sha(meta_path)
        for filename, label in labels.items():
            if label not in ('real', 'fake'):
                raise ValueError(f'Chưa gán nhãn / nhãn sai: {labels_path}:{filename}')
            if Path(filename).name != filename:
                raise ValueError('Crop key must be a filename')
            path = labels_path.parent / filename
            crop = cv2.imread(str(path))
            if crop is None:
                raise ValueError(f'Không đọc được {path}')
            digest = sha(path)
            if digest in known and known[digest] != label:
                raise ValueError(f'Conflicting duplicate labels: {path}')
            known[digest] = label
            meta = metadata[filename]
            if meta['crop_mode'] != 'minifasnet_train_v1' or meta['lighting_adjusted']:
                raise ValueError(f'Unsupported capture contract: {path}')
            samples.append(dict(file=str(path.resolve()), session=path.parent.name, label=label,
                                sha256=digest, metadata=meta, crop=crop))
    if set(s['label'] for s in samples) != {'real', 'fake'}:
        raise ValueError('Cần cả real và fake')
    return samples, snapshots


def reconstruct_crop(sample, factor):
    meta, crop = sample['metadata'], sample['crop']
    frame_file = meta.get('frame_file')
    if frame_file:
        frame = cv2.imread(str(Path(sample['file']).parent / frame_file))
        if frame is None:
            raise ValueError(f'Missing source frame: {frame_file}')
        return crop_for_runtime(frame, meta['bbox_xyxy'], factor, 80, meta['crop_mode'])
    if factor > meta['bbox_expansion_factor']:
        raise ValueError('Cannot expand beyond captured crop without source frame')
    source, _ = expanded_bounds(meta['frame_shape'], meta['bbox_xyxy'], meta['bbox_expansion_factor'])
    target, _ = expanded_bounds(meta['frame_shape'], meta['bbox_xyxy'], factor)
    sx, sy, ex, ey = source
    tx, ty, rx, by = target
    if crop.shape[:2] != (ey - sy + 1, ex - sx + 1):
        raise ValueError(f'Native crop geometry differs: {sample["file"]}')
    if not (sx <= tx <= rx <= ex and sy <= ty <= by <= ey):
        raise ValueError('Desired ROI not contained in saved native crop')
    patch = crop[ty-sy:by-sy+1, tx-sx:rx-sx+1]
    return cv2.resize(patch, (80, 80))


def profiles(refine=False):
    if refine:
        return ([{'method': 'none'}]
                + [dict(method=method, gamma=round(g, 2)) for method in ('gamma', 'gamma_y')
                   for g in np.arange(.4, .801, .05)]
                + [dict(method='gamma_clahe_y', gamma=g, clip=c, tile=4, blend=.25)
                   for g in (.55,.65,.75) for c in (1.2,2.)])
    result = [{'method': 'none'}, {'method': 'legacy_gamma'}]
    result += [dict(method='gamma', gamma=g) for g in (.65, .8, .9, 1.1, 1.25, 1.5)]
    result += [dict(method='adaptive_gamma', target=t, min_gamma=lo, max_gamma=hi, deadband=d)
               for t in (95, 110, 125) for lo,hi in ((.7,1.3),(.85,1.15)) for d in (10,20)]
    result += [dict(method='gain_y', target=t, min_gain=.8, max_gain=1.3) for t in (95,110,125)]
    result += [dict(method='clahe_y', clip=c, tile=t, blend=b)
               for c in (1.2,2.,3.) for t in (4,8) for b in (.25,.5)]
    result += [dict(method='stretch_y', low_percentile=1, high_percentile=99,max_gain=g,blend=.5)
               for g in (1.2,1.5,2.)]
    return result


def fit(scores, labels):
    best, _ = select_threshold(scores, labels)
    t = best['threshold_logit']
    lower = np.asarray(scores)[np.asarray(scores) < t]
    upper = np.asarray(scores)[np.asarray(scores) >= t]
    # Midpoint of the decision-equivalent interval avoids equality at a sample.
    if lower.size and upper.size:
        t = (float(lower.max()) + float(upper.min())) / 2
    return metrics(scores, labels, t)


def perturb(image, name):
    if name == 'original':
        return image
    if name.startswith('gain'):
        return np.clip(image.astype(np.float32)*float(name.split('_')[1]),0,255).astype(np.uint8)
    if name.startswith('gamma'):
        return LightingTransform(dict(method='gamma',gamma=float(name.split('_')[1])))(image)
    if name == 'jpeg90':
        ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
        assert ok
        return cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if name == 'blur':
        return cv2.GaussianBlur(image,(3,3),.4)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'pad_captures')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refine',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    samples,snapshots=load_samples(args.data)
    y=np.array([int(s['label']=='real') for s in samples])
    model=Path(config.PAD_MODEL_PATH)
    assert sha(model)==config.PAD_RUNTIME_MODEL_SHA256
    assert config.PAD_MODEL_IMG_SIZE==80 and config.PAD_COLOR_ORDER=='BGR'
    assert config.PAD_MEAN is None and config.PAD_STD is None
    assert not config.PAD_GAMMA_ENABLED, 'This grid assumes gamma-off source baseline'
    assert config.PAD_LIGHTING_PROFILE is None and config.PAD_ADAPTIVE_THRESHOLD_PROFILE is None, 'Run from the original source runtime, not a fitted candidate'
    assert config.PAD_BBOX_EXPANSION_FACTOR == 1.5, 'Source baseline requires crop factor 1.5'
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name
    input_batch=session.get_inputs()[0].shape[0]
    batch_size=1 if isinstance(input_batch,int) else 16
    def score(images):
        predictions=[]
        for start in range(0,len(images),batch_size):
            tensor=preprocess_batch(images[start:start+batch_size],80,mean=None,std=None,apply_gamma=False,convert_rgb=False)
            logits=session.run(None,{input_name:tensor})[0]
            predictions.extend((logits[:,0]-logits[:,1]).astype(float).tolist())
        values=np.asarray(predictions)
        assert len(values)==len(images) and np.isfinite(values).all()
        return values
    manifest=[{k:s[k] for k in ('file','session','label','sha256')} for s in samples]
    write_csv(args.output/'manifest.csv',manifest)
    provenance=dict(run_class='exploratory_all_data_fit',heldout=False,seed=20261021,
        created_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        model=str(model),model_sha256=sha(model),runtime_config=config.PAD_RUNTIME_CONFIG_PATH,
        runtime_sha256=sha(Path(config.PAD_RUNTIME_CONFIG_PATH)),
        source_sha256=sha(Path(__file__)),lighting_sha256=sha(ROOT/'antispoof/lighting.py'),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.version,platform=platform.platform(),opencv=cv2.__version__,onnxruntime=ort.__version__,
        providers=session.get_providers(),intra_op_threads=1,batch_size=batch_size,
        counts=dict(Counter(s['label'] for s in samples)),unique_hashes=len(set(s['sha256'] for s in samples)),
        snapshots=snapshots)
    write_json(args.output/'provenance.json',provenance)
    (args.output/'experiment_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'lighting_source.py').write_bytes((ROOT/'antispoof/lighting.py').read_bytes())
    base=[reconstruct_crop(s,config.PAD_BBOX_EXPANSION_FACTOR) for s in samples]
    for s,image in zip(samples,base):
        saved_input=cv2.imread(str(Path(s['file']).parent/s['metadata']['input_file']))
        if s['metadata']['bbox_expansion_factor']==config.PAD_BBOX_EXPANSION_FACTOR:
            np.testing.assert_array_equal(image,saved_input)
    # Warm up before grid timing; inference includes preprocessing but not transforms.
    score(base[:min(16,len(base))])
    baseline_scores=score(base)
    baseline=metrics(baseline_scores,y,config.PAD_THRESHOLD_LOGIT)
    results=[]; rows=[]; score_cache={}; crop_cache={}; configs={}
    factors = [round(v,3) for v in np.arange(1.25,1.501,.025)] if args.refine else (1.,1.1,1.2,1.3,1.4,1.5)
    for factor in factors:
        crops=[reconstruct_crop(s,factor) for s in samples]
        crop_cache[factor]=crops
        for idx,profile in enumerate(profiles(args.refine)):
            factor_label = f'{factor:.3f}' if args.refine else f'{factor:.1f}'
            name=f'crop{factor_label}_{idx:02d}_{profile["method"]}'
            transform=LightingTransform(profile)
            started=time.perf_counter();images=[transform(c) for c in crops]
            elapsed=(time.perf_counter()-started)*1000/len(images)
            started=time.perf_counter();scores=score(images)
            inference_ms=(time.perf_counter()-started)*1000/len(images)
            result=fit(scores,y)
            result.update(candidate=name,crop_factor=factor,profile=json.dumps(profile,sort_keys=True),
                          transform_mean_ms=elapsed,inference_with_preprocess_mean_ms=inference_ms)
            results.append(result); score_cache[name]=scores;configs[name]=(factor,profile)
            rows.extend(dict(candidate=name,file=s['file'],session=s['session'],label=s['label'],score_logit=float(d))
                        for s,d in zip(samples,scores))
        print(f'Finished crop={factor:.3f}: {len(results)} candidates',flush=True)
        write_csv(args.output/'candidates.csv',results)
    write_csv(args.output/'scores.csv',rows)
    ranked=sorted(results,key=lambda r:(r['ACER'],r['APCER'],r['transform_mean_ms']))
    # Include best per method, overall top 12 and all crop-only candidates.
    shortlist={r['candidate'] for r in ranked[:12]}
    for method in {p['method'] for p in profiles(args.refine)}:
        shortlist.add(next(r['candidate'] for r in ranked if json.loads(r['profile'])['method']==method))
    shortlist.update(r['candidate'] for r in results if json.loads(r['profile'])['method']=='none')
    robustness=[]
    perturbations=('original','gain_0.9','gain_1.1','gamma_0.95','gamma_1.05','jpeg90','blur')
    for name in sorted(shortlist):
        factor,profile=configs[name];transform=LightingTransform(profile)
        threshold=next(r['threshold_logit'] for r in results if r['candidate']==name)
        clean=score_cache[name]>=threshold
        for change in perturbations:
            scores=score_cache[name] if change=='original' else score([transform(perturb(c,change)) for c in crop_cache[factor]])
            m=metrics(scores,y,threshold)
            m.update(candidate=name,perturbation=change,flip_rate=float(np.mean((scores>=threshold)!=clean)))
            robustness.append(m)
        print(f'Robustness {name}',flush=True)
    write_csv(args.output/'robustness.csv',robustness)
    for r in results:
        entries=[v for v in robustness if v['candidate']==r['candidate']]
        if entries:
            r['worst_perturbation_ACER']=max(e['ACER'] for e in entries)
            r['mean_perturbation_ACER']=float(np.mean([e['ACER'] for e in entries]))
            r['mean_flip_rate']=float(np.mean([e['flip_rate'] for e in entries]))
    eligible=[r for r in results if r['candidate'] in shortlist]
    recommended=min(eligible,key=lambda r:(r['ACER'],r['APCER'],r['worst_perturbation_ACER'],r['mean_perturbation_ACER'],
                                          json.loads(r['profile'])['method']!='none',r['transform_mean_ms']))
    threshold_only=next(r for r in results if r['crop_factor']==config.PAD_BBOX_EXPANSION_FACTOR
                        and json.loads(r['profile'])['method']=='none')
    report=dict(provenance=provenance,baseline=baseline,threshold_only=threshold_only,
                best_fit=ranked[0],recommended=recommended,candidates=len(results),
                limitations=['All-data fitted; no held-out evaluation','Synthetic perturbations are not temporal validation',
                             'Crop >1.5 not available on current captures','Latency measured on local host, not Raspberry Pi'])
    write_json(args.output/'report.json',report)
    # Rectangular crop reconstruction is exact; profile operates after MiniFASNet resize.
    for export_name,result in (('candidate_runtime.json',recommended),('threshold_only_runtime.json',threshold_only)):
        runtime=json.loads(Path(config.PAD_RUNTIME_CONFIG_PATH).read_text())
        runtime.update(model_file=str(model),bbox_expansion_factor=result['crop_factor'],apply_gamma=False,
                       lighting=json.loads(result['profile']),calibrated_logit_threshold=result['threshold_logit'],
                       predictor_threshold_probability=probability(result['threshold_logit']),
                       threshold_source='All local capture data fit; no held-out; exploratory camera candidate',
                       threshold_calibration=dict(report=str((args.output/'report.json').resolve()),heldout=False))
        write_json(args.output/export_name,runtime)
    chosen_scores=score_cache[recommended['candidate']]
    write_csv(args.output/'recommended_predictions.csv',[
        dict(file=s['file'],session=s['session'],label=s['label'],score_logit=float(d),
             predicted='real' if d>=recommended['threshold_logit'] else 'fake',
             correct=bool((d>=recommended['threshold_logit'])==bool(label)))
        for s,d,label in zip(samples,chosen_scores,y)])
    _,curve=select_threshold(chosen_scores,y);write_csv(args.output/'recommended_threshold_sweep.csv',curve)
    for path,digest in snapshots.items():
        assert sha(Path(path))==digest, f'Data changed while running: {path}'
    for s in samples:
        assert sha(Path(s['file']))==s['sha256']
    print(json.dumps(dict(baseline=baseline,threshold_only=threshold_only,recommended=recommended),indent=2),flush=True)


if __name__=='__main__':
    main()
