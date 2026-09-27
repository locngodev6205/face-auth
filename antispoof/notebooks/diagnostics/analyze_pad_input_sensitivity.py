"""Offline PAD input sensitivity; unlabeled images do not measure accuracy.

Uses the active runtime contract. It does not change runtime or calibrate a threshold.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import config
from antispoof.predictor import AntiSpoofPredictor
from antispoof.preprocess import crop
from detection.detector import FaceDetector


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predictor = AntiSpoofPredictor()
    if predictor.is_frequency_model:
        raise ValueError("This sensitivity probe is for the active spatial E1 contract")
    detector = FaceDetector(model_name=config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID,
                            det_size=config.DETECTOR_DET_SIZE,
                            conf_thresh=config.DETECTOR_CONF_THRESH,
                            min_face_size=config.DETECTOR_MIN_FACE_SIZE)
    samples = []
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read {path}")
        detections = detector.detect(image)
        if not detections:
            samples.append({"image": str(path), "sha256": sha256(path), "status": "NO_FACE"})
            continue
        det = max(detections, key=lambda d: (d["bbox"][2]-d["bbox"][0]) *
                  (d["bbox"][3]-d["bbox"][1]))
        bbox = np.asarray(det["bbox"], dtype=float)
        x1, y1, x2, y2 = bbox
        side = max(x2-x1, y2-y1)
        variants = [("baseline", image, bbox)]
        for axis in [0, 1]:
            for sign in [-1, 1]:
                moved = bbox.copy()
                moved[[axis, axis+2]] += sign * 0.05 * side
                variants.append((f"shift_{axis}_{sign}_5pct", image, moved))
        center = np.array([(x1+x2)/2, (y1+y2)/2] * 2)
        for factor in [0.95, 1.05]:
            variants.append((f"bbox_scale_{factor}", image, center+(bbox-center)*factor))
        # Fixed bbox association isolates image sampling from detector behavior.
        for scale in [0.75, 0.5]:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                                 interpolation=cv2.INTER_AREA)
            actual_scale = np.array([resized.shape[1]/w, resized.shape[0]/h] * 2)
            variants.append((f"frame_downsample_{scale}", resized, bbox*actual_scale))
        crops = [crop(img, tuple(box), predictor.bbox_expansion_factor) for _, img, box in variants]
        results = predictor.predict_crops(crops)
        if len(results) != len(variants):
            raise RuntimeError("PAD inference failed")
        # Check whether batch shape/repeated inference changes the baseline score.
        singleton = predictor.predict_crops([crops[0]])
        repeated = predictor.predict_crops(crops)
        if len(singleton) != 1 or len(repeated) != len(results):
            raise RuntimeError("PAD repeat/singleton inference failed")
        samples.append({
            "image": str(path), "sha256": sha256(path), "status": "OK",
            "image_shape": list(image.shape), "detector_bbox": bbox.tolist(),
            "inference_checks": {
                "batch_vs_singleton_baseline_abs_score_delta": abs(
                    results[0]["pad_score"] - singleton[0]["pad_score"]),
                "repeated_batch_max_abs_score_delta": max(
                    abs(a["pad_score"] - b["pad_score"])
                    for a, b in zip(results, repeated)),
                "batch_vs_singleton_baseline_verdict_equal":
                    results[0]["is_real"] == singleton[0]["is_real"],
            },
            "variants": [dict(name=name, bbox=box.tolist(),
                              pad_score=res["pad_score"], p_real=res["class_probs"]["real"],
                              is_real=res["is_real"])
                         for (name, _, box), res in zip(variants, results)],
        })
    output = {
        "scope": "Unlabeled image sensitivity, not accuracy or camera benchmark",
        "model": str(predictor.model_path), "model_sha256": sha256(predictor.model_path),
        "external_data_sha256": sha256(str(predictor.model_path) + ".data")
            if Path(str(predictor.model_path) + ".data").is_file() else None,
        "provider": list(predictor.providers),
        "contract": dict(size=predictor.model_img_size, expansion=predictor.bbox_expansion_factor,
                         color=predictor.color_order, gamma=predictor.apply_gamma,
                         mean=predictor.mean, std=predictor.std, threshold=predictor.logit_threshold),
        "sampling_policy": "largest detected face; fixed bbox for frame degradation",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {len(samples)} image sensitivity records to {args.output}")


if __name__ == "__main__":
    main()
