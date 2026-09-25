#!/usr/bin/env python3
"""
camera_pad_diagnostic.py

Local/runtime diagnostic for a trained Face PAD ONNX model.

Designed to answer:
1) Is the correct model/runtime config loaded?
2) Is the probability threshold being passed with the correct semantics?
3) Does gamma cause the Real->Spoof collapse?
4) Is there a BGR/RGB double-conversion bug?
5) Is the result extremely sensitive to bbox expansion / detector geometry?
6) If the correct runtime pipeline still predicts camera Real as Spoof,
   export scores for CelebA-vs-camera domain-shift comparison in Kaggle.

Expected project layout:
    from antispoof.predictor import AntiSpoofPredictor
    from antispoof.preprocess import crop

OpenCV camera frames are BGR. For MobileNetV3, AntiSpoofPredictor should perform
the BGR->RGB conversion internally when color_order="RGB".
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from antispoof.predictor import AntiSpoofPredictor
    from antispoof.preprocess import crop
except ImportError as exc:
    raise SystemExit(
        "Cannot import antispoof package. Run this script from the repository root "
        "or set PYTHONPATH=. Example:\n"
        "  PYTHONPATH=. python camera_pad_diagnostic.py ...\n"
        f"Original error: {exc}"
    )

CROP_FACTORS = [1.0, 1.25, 1.5, 1.75, 2.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to trained MobileNetV3 ONNX")
    p.add_argument("--runtime-config", required=True, help="Path to *_runtime_config.json")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="OpenCV camera index, e.g. 0")
    src.add_argument("--image", help="Single BGR image file")
    p.add_argument(
        "--bbox",
        help="Optional xyxy bbox: x1,y1,x2,y2. If omitted, select ROI interactively."
    )
    p.add_argument("--output-dir", default="camera_pad_diagnostic_output")
    p.add_argument("--max-snapshots", type=int, default=30)
    return p.parse_args()


def bbox_from_text(text):
    if not text:
        return None
    vals = [int(float(x.strip())) for x in text.split(",")]
    if len(vals) != 4:
        raise ValueError("--bbox must be x1,y1,x2,y2")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid xyxy bbox")
    return (x1, y1, x2, y2)


def select_bbox(frame):
    x, y, w, h = cv2.selectROI(
        "Select FACE bbox then press ENTER", frame, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select FACE bbox then press ENTER")
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled/invalid")
    return (int(x), int(y), int(x+w), int(y+h))


def get_probability_threshold(cfg):
    for key in [
        "predictor_threshold_probability",
        "calibrated_probability_threshold",
        "threshold_probability",
    ]:
        if key in cfg:
            return float(cfg[key])
    raise KeyError("Runtime config has no predictor probability threshold")


def get_logit_threshold(cfg):
    for key in ["calibrated_logit_threshold", "logit_threshold"]:
        if key in cfg:
            return float(cfg[key])
    p = get_probability_threshold(cfg)
    p = min(1-1e-6, max(1e-6, p))
    return float(math.log(p/(1-p)))


def build_predictor(model, cfg, *, threshold=None, gamma=None, crop_factor=None):
    if threshold is None:
        threshold = get_probability_threshold(cfg)
    if gamma is None:
        gamma = bool(cfg.get("apply_gamma", False))
    if crop_factor is None:
        crop_factor = float(cfg.get("bbox_expansion_factor", 1.5))

    return AntiSpoofPredictor(
        model_path=str(model),
        threshold=float(threshold),
        model_img_size=int(cfg.get("model_img_size", 224)),
        bbox_expansion_factor=float(crop_factor),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        apply_gamma=bool(gamma),
        color_order=cfg.get("color_order", "RGB"),
    )


def compact_result(result):
    return {
        "status": result.get("status"),
        "is_real": bool(result.get("is_real", False)),
        "pad_score": float(result.get("pad_score", np.nan)),
        "p_real": float(result.get("class_probs", {}).get("real", np.nan)),
        "confidence": float(result.get("confidence", np.nan)),
        "detailed_status": result.get("detailed_status"),
    }


def diagnostic_predictors(model, cfg):
    correct_p = get_probability_threshold(cfg)
    pred_correct = build_predictor(model, cfg, threshold=correct_p, gamma=False)
    pred_gamma = build_predictor(model, cfg, threshold=correct_p, gamma=True)
    pred_default_threshold = build_predictor(model, cfg, threshold=0.5, gamma=False)

    expected_logit = get_logit_threshold(cfg)
    actual = float(pred_correct.logit_threshold)
    print("\n=== Runtime contract ===")
    print("Model:", pred_correct.model_path)
    print("Input size:", pred_correct.model_img_size)
    print("Color order:", pred_correct.color_order)
    print("Mean:", pred_correct.mean)
    print("Std:", pred_correct.std)
    print("Research gamma:", False)
    print("Probability threshold:", correct_p)
    print("Expected logit threshold:", expected_logit)
    print("Predictor logit threshold:", actual)

    if abs(actual - expected_logit) > 1e-5:
        raise RuntimeError(
            "THRESHOLD SEMANTIC MISMATCH. "
            "Do not pass a logit threshold into AntiSpoofPredictor(threshold=...)."
        )
    return pred_correct, pred_gamma, pred_default_threshold


def run_snapshot(frame_bgr, bbox, snapshot_id, out_dir, predictors, cfg):
    pred_correct, pred_gamma, pred_default = predictors

    # Correct research runtime: OpenCV BGR frame -> predictor converts internally.
    correct = compact_result(pred_correct.predict_frame(frame_bgr, bbox))

    # A/B test gamma mismatch.
    gamma = compact_result(pred_gamma.predict_frame(frame_bgr, bbox))

    # Simulate the common bug: caller pre-converts BGR->RGB, then predictor swaps again.
    frame_rgb_wrong = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    wrong_rgb = compact_result(pred_correct.predict_frame(frame_rgb_wrong, bbox))

    # Compare calibrated threshold against default 0.5.
    default_t = compact_result(pred_default.predict_frame(frame_bgr, bbox))

    # Bbox/crop sensitivity using the SAME correctly configured model/session.
    crop_scores = {}
    for factor in CROP_FACTORS:
        face_crop_bgr = crop(frame_bgr, bbox, factor)
        res = compact_result(pred_correct.predict_crops([face_crop_bgr])[0])
        crop_scores[factor] = res

    row = {
        "snapshot_id": snapshot_id,
        "timestamp": time.time(),
        "bbox_x1": bbox[0], "bbox_y1": bbox[1],
        "bbox_x2": bbox[2], "bbox_y2": bbox[3],
        "locked_probability_threshold": get_probability_threshold(cfg),
        "locked_logit_threshold": get_logit_threshold(cfg),
        "pad_score_correct": correct["pad_score"],
        "p_real_correct": correct["p_real"],
        "status_correct": correct["status"],
        "pad_score_gamma": gamma["pad_score"],
        "p_real_gamma": gamma["p_real"],
        "status_gamma": gamma["status"],
        "pad_score_wrong_rgb": wrong_rgb["pad_score"],
        "p_real_wrong_rgb": wrong_rgb["p_real"],
        "status_wrong_rgb": wrong_rgb["status"],
        "pad_score_default_threshold": default_t["pad_score"],
        "p_real_default_threshold": default_t["p_real"],
        "status_default_threshold": default_t["status"],
    }
    for factor in CROP_FACTORS:
        key = str(factor).replace(".", "_")
        row[f"crop_{key}_score"] = crop_scores[factor]["pad_score"]
        row[f"crop_{key}_status"] = crop_scores[factor]["status"]

    scores = np.array([crop_scores[f]["pad_score"] for f in CROP_FACTORS], dtype=float)
    row["crop_score_range"] = float(np.ptp(scores))
    row["crop_score_std"] = float(np.std(scores))

    # Save raw frame + canonical 1.5x crop for later inspection.
    frame_file = out_dir / f"snapshot_{snapshot_id:03d}_frame.jpg"
    crop_file = out_dir / f"snapshot_{snapshot_id:03d}_crop15.jpg"
    overlay_file = out_dir / f"snapshot_{snapshot_id:03d}_overlay.jpg"
    cv2.imwrite(str(frame_file), frame_bgr)

    canonical_crop = crop(frame_bgr, bbox, float(cfg.get("bbox_expansion_factor", 1.5)))
    cv2.imwrite(str(crop_file), canonical_crop)

    overlay = frame_bgr.copy()
    x1,y1,x2,y2 = bbox
    cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(
        overlay,
        f"correct={correct['status']} score={correct['pad_score']:.3f}",
        (max(0,x1), max(25,y1-8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2
    )
    cv2.imwrite(str(overlay_file), overlay)

    print("\n=== Snapshot", snapshot_id, "===")
    print("Correct BGR/no-gamma :", correct)
    print("Gamma=True           :", gamma)
    print("Wrong double RGB swap:", wrong_rgb)
    print("Default threshold 0.5:", default_t)
    print("Crop factor sweep:")
    for factor in CROP_FACTORS:
        r = crop_scores[factor]
        print(f"  {factor:>4}: score={r['pad_score']:+.4f}  status={r['status']}")
    print("Crop score range/std:", row["crop_score_range"], row["crop_score_std"])

    # Lightweight interpretation; final diagnosis should use the guide + multiple samples.
    t = get_logit_threshold(cfg)
    if correct["pad_score"] >= t:
        print("✓ Correct pipeline accepts this frame as Real.")
        print("  If the app still says Spoof, inspect app integration/threshold/result handling.")
    else:
        if gamma["pad_score"] < correct["pad_score"] - 1.0:
            print("! Gamma materially lowers the Real score: possible gamma preprocessing mismatch.")
        if wrong_rgb["pad_score"] < correct["pad_score"] - 1.0:
            print("! Double RGB/BGR conversion materially lowers score: inspect caller color conversion.")
        if row["crop_score_range"] >= 2.0:
            print("! Strong bbox/crop sensitivity: inspect detector geometry and crop factor.")
        if (
            gamma["pad_score"] < t
            and wrong_rgb["pad_score"] < t
            and max(scores) < t
        ):
            print(
                "! All tested runtime variants remain Spoof. "
                "If CelebA evaluation is healthy, camera-domain shift becomes a strong candidate."
            )

    return row


def append_csv(csv_path, row):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def image_mode(args, predictors, cfg, out_dir):
    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(args.image)
    bbox = bbox_from_text(args.bbox) or select_bbox(frame)
    row = run_snapshot(frame, bbox, 1, out_dir, predictors, cfg)
    append_csv(out_dir / "camera_pad_diagnostic.csv", row)


def camera_mode(args, predictors, cfg, out_dir):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    snapshot_id = 0
    print("\nCamera controls:")
    print("  SPACE = freeze frame, select face ROI, run diagnostic")
    print("  q     = quit")

    try:
        while snapshot_id < args.max_snapshots:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            cv2.putText(
                view, "SPACE: diagnose snapshot | q: quit",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2
            )
            cv2.imshow("PAD Camera Diagnostic", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # SPACE
                bbox = bbox_from_text(args.bbox) or select_bbox(frame)
                snapshot_id += 1
                row = run_snapshot(
                    frame.copy(), bbox, snapshot_id, out_dir, predictors, cfg
                )
                append_csv(out_dir / "camera_pad_diagnostic.csv", row)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = Path(args.model)
    cfg_path = Path(args.runtime_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model.exists():
        raise FileNotFoundError(model)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    predictors = diagnostic_predictors(model, cfg)

    if args.image:
        image_mode(args, predictors, cfg, out_dir)
    else:
        camera_mode(args, predictors, cfg, out_dir)

    print("\nOutput:", out_dir.resolve())
    print("CSV:", (out_dir / "camera_pad_diagnostic.csv").resolve())
    print(
        "Upload this CSV to Kaggle and use Cell 19 of "
        "pad_post_training_deep_evaluation_kaggle.ipynb "
        "to compare Camera Real vs CelebA Real score distributions."
    )


if __name__ == "__main__":
    main()
