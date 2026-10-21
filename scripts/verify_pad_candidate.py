"""Reload exported runtime with the actual Predictor and check fitted scores."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-json',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--adaptive-predictions',type=Path)
    args=parser.parse_args()
    os.environ['PAD_RUNTIME_CONFIG_PATH']=str(args.runtime_json.resolve())
    for key in ('PAD_THRESHOLD','PAD_THRESHOLD_LOGIT','PAD_BBOX_EXPANSION_FACTOR'):
        os.environ[key]=''
    os.environ['FORCE_GAMMA']='false'
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import config
    from antispoof.predictor import AntiSpoofPredictor
    from optimize_pad_domain import load_samples,reconstruct_crop,write_json
    samples,_=load_samples(root/'pad_captures')
    expected={}
    with args.scores.open() as stream:
        for row in csv.DictReader(stream):
            if row['candidate']==args.candidate and row.get('perturbation','original')=='original':
                expected[row['file']]=float(row['score_logit'])
    assert set(expected)=={s['file'] for s in samples}
    predictor=AntiSpoofPredictor()
    actual=[];decisions=[];thresholds=[]
    for start in range(0,len(samples),16):
        crops=[reconstruct_crop(s,predictor.bbox_expansion_factor) for s in samples[start:start+16]]
        results=predictor.predict_crops(crops)
        assert len(results)==len(crops)
        actual.extend(float(r['pad_score']) for r in results)
        decisions.extend(r['is_real'] for r in results)
        thresholds.extend(r['threshold_logit'] for r in results)
    expected_scores=np.array([expected[s['file']] for s in samples]);actual=np.array(actual)
    maximum=float(np.max(np.abs(actual-expected_scores)))
    assert maximum<1e-4, maximum
    if args.adaptive_predictions:
        with args.adaptive_predictions.open() as stream:
            expected_decisions={r['file']:r for r in csv.DictReader(stream)}
        expected_thresholds=np.array([float(expected_decisions[s['file']]['effective_threshold']) for s in samples])
        np.testing.assert_allclose(thresholds,expected_thresholds,rtol=0,atol=1e-9)
    else:
        expected_thresholds=np.full(len(samples),predictor.logit_threshold)
    np.testing.assert_array_equal(decisions,expected_scores>=expected_thresholds)
    result=dict(n_images=len(samples),max_absolute_score_error=maximum,
                decision_match=True,threshold=predictor.logit_threshold,
                crop_factor=predictor.bbox_expansion_factor,lighting=predictor.lighting_transform.profile,
                adaptive_threshold=predictor.threshold_adapter.profile,
                providers=predictor.providers)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
