"""Hiệu chỉnh PAD threshold từ thư mục chứa real/ và fake/ (đọc đệ quy).

Đầu vào là ẢNH MẶT ĐÃ CROP 1.5x từ capture_domain_faces.py, không phải frame.
Không detect/crop lần nữa. MiniFASNet resize trực tiếp trước preprocessing runtime.
Ví dụ từ repo root:
  python3 face_auth/scripts/calibrate_domain_threshold.py --data /path/calibration
  ... --data /path/calibration --eval-data /path/heldout
  ... --runtime-config face_auth/antispoof/models/R7_SC_100K_crop15_runtime_config.json

ACER pooled binary = HTER = (APCER + BPCER)/2, rates nằm trong [0,1].
Không phải ACER dùng worst-case APCER theo từng loại tấn công.
Đây là threshold theo ảnh, chưa đánh giá temporal voting của app camera.
Dùng người/phiên riêng cho calibration và held-out; không fit trên official test.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def probability(logit):
    if logit >= 0:
        return 1 / (1 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1 + exp_value)


def metrics(scores, labels, threshold):
    scores, labels = validate_scores(scores, labels)
    if not math.isfinite(threshold):
        raise ValueError("Threshold phải hữu hạn")
    predicted_real = scores >= threshold
    real = labels == 1
    n_real, n_fake = int(real.sum()), int((~real).sum())
    fake_as_real = int((predicted_real & ~real).sum())
    real_as_fake = int((~predicted_real & real).sum())
    apcer, bpcer = fake_as_real / n_fake, real_as_fake / n_real
    return {
        "threshold_logit": float(threshold), "threshold_probability": probability(threshold),
        "n_real": n_real, "n_fake": n_fake,
        "fake_as_real": fake_as_real, "real_as_fake": real_as_fake,
        "APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2,
        "FAR": apcer, "FRR": bpcer, "HTER": (apcer + bpcer) / 2,
    }


def validate_scores(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or scores.size == 0 or labels.shape != scores.shape:
        raise ValueError("Scores/labels phải là vector cùng kích thước, không rỗng")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Score phải hữu hạn; label real=1, fake=0")
    if not (np.any(labels == 1) and np.any(labels == 0)):
        raise ValueError("Phải có ít nhất một ảnh real và một ảnh fake")
    return scores, labels.astype(np.int64)


def select_threshold(scores, labels):
    """Exact O(N log N) sweep, including tied scores and both end decisions.

    At threshold t, real iff score >= t. Candidates are every unique score
    and nextafter(max_score, +inf), covering every possible prediction vector.
    Equal minima: lowest APCER, then largest threshold. Integer objective
    avoids floating-point errors when comparing balanced error rates.
    """
    scores, labels = validate_scores(scores, labels)
    order = np.argsort(scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    thresholds = np.unique(sorted_scores)
    upper = float(np.nextafter(thresholds[-1], np.inf))
    if not math.isfinite(upper):
        raise ValueError("Score vượt miền threshold hữu hạn được hỗ trợ")
    thresholds = np.append(thresholds, upper)
    below = np.searchsorted(sorted_scores, thresholds, side="left")
    real_prefix = np.concatenate(([0], np.cumsum(sorted_labels)))
    n_real, n_fake = int(labels.sum()), int(len(labels) - labels.sum())
    real_as_fake = real_prefix[below]
    fake_as_real = n_fake - (below - real_as_fake)
    objective = real_as_fake * n_fake + fake_as_real * n_real
    best = min(range(len(thresholds)),
               key=lambda i: (int(objective[i]), int(fake_as_real[i]), -float(thresholds[i])))
    curve = []
    for t, false_accepts, false_rejects in zip(thresholds, fake_as_real, real_as_fake):
        apcer, bpcer = int(false_accepts) / n_fake, int(false_rejects) / n_real
        curve.append({"threshold_logit": float(t), "threshold_probability": probability(float(t)),
                      "fake_as_real": int(false_accepts), "real_as_fake": int(false_rejects),
                      "APCER": apcer, "BPCER": bpcer, "ACER": (apcer + bpcer) / 2,
                      "HTER": (apcer + bpcer) / 2})
    return metrics(scores, labels, float(thresholds[best])), curve


def collect_images(directory):
    directory = directory.expanduser().resolve()
    files = []
    for name, label in (("real", 1), ("fake", 0)):
        folder = directory / name
        if not folder.is_dir():
            raise ValueError(f"Thiếu thư mục {folder}")
        found = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not found:
            raise ValueError(f"Không có ảnh trong {folder}")
        files.extend((p, label) for p in found)
    return directory, files


def score_images(directory, files, predictor, batch_size):
    rows = []
    # A model with fixed batch=1 cannot accept the CLI's larger batch default.
    model_batch = predictor.input_metadata.shape[0]
    if isinstance(model_batch, int):
        if model_batch != 1:
            raise ValueError("Chỉ hỗ trợ model dynamic batch hoặc fixed batch=1")
        batch_size = 1
    for start in range(0, len(files), batch_size):
        patches, batch_rows = [], []
        for path, label in files[start:start + batch_size]:
            data = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError(f"Ảnh hỏng/không đọc được: {path}; dừng, không bỏ qua mẫu")
            if predictor.crop_mode == "minifasnet_train_v1":
                # Runtime crop_minifasnet_train_v1 uses INTER_LINEAR BEFORE
                # predict_crops' letterbox. Native rectangular captures need this too.
                image = cv2.resize(image, (predictor.model_img_size, predictor.model_img_size))
            patches.append(image)
            batch_rows.append({"file": path.relative_to(directory).as_posix(),
                               "label": "real" if label else "fake",
                               "sha256": hashlib.sha256(data).hexdigest()})
        results = predictor.predict_crops(patches)
        if len(results) != len(batch_rows):
            raise RuntimeError(f"Suy luận thất bại tại batch {start}; không tạo threshold từ kết quả thiếu")
        for row, result in zip(batch_rows, results):
            score = float(result["pad_score"])
            if not math.isfinite(score):
                raise ValueError(f"Score không hữu hạn: {row['file']}")
            row.update(score_logit=score, probability_real=probability(score))
            rows.append(row)
        print(f"  {len(rows)}/{len(files)} ảnh", flush=True)
    # Identical bytes under conflicting labels cannot be meaningful calibration data.
    known = {}
    for row in rows:
        if row["sha256"] in known and known[row["sha256"]] != row["label"]:
            raise ValueError(f"Ảnh trùng nội dung trong cả real/fake: {row['file']}")
        known[row["sha256"]] = row["label"]
    return rows


def row_arrays(rows):
    return [row["score_logit"] for row in rows], [int(row["label"] == "real") for row in rows]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_runtime(source, predictor, best, provenance):
    """Copy effective preprocessing and threshold, retaining the model contract."""
    threshold, p = best["threshold_logit"], best["threshold_probability"]
    if not 0 < p < 1 or abs(math.log(p / (1 - p)) - threshold) > 1e-6:
        raise ValueError("Threshold tối ưu vượt độ chính xác probability/logit mà config.py hỗ trợ; xem report.json")
    runtime = dict(source)
    runtime.update({
        "model_file": str(predictor.model_path),
        "model_img_size": predictor.model_img_size,
        "bbox_expansion_factor": predictor.bbox_expansion_factor,
        "crop_mode": predictor.crop_mode, "apply_gamma": predictor.apply_gamma,
        "color_order": predictor.color_order, "mean": predictor.mean, "std": predictor.std,
        "calibrated_logit_threshold": threshold, "predictor_threshold_probability": p,
        "decision_rule": "real iff score >= calibrated_logit_threshold",
        "threshold_source": "Local labeled domain calibration images; minimum pooled binary ACER/HTER; image-level only",
        "threshold_calibration": provenance,
    })
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, type=Path, help="Tập hiệu chỉnh có real/ và fake/")
    parser.add_argument("--eval-data", type=Path, help="Tập đánh giá độc lập có real/ và fake/; KHÔNG fit lại")
    parser.add_argument("--runtime-config", type=Path, help="Mặc định dùng PAD_RUNTIME_CONFIG_PATH trong .env")
    parser.add_argument("--output", type=Path, default=ROOT / "output/domain_threshold")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size phải >= 1")
    calibration_dir, calibration_files = collect_images(args.data)
    evaluation = collect_images(args.eval_data) if args.eval_data else None
    if args.runtime_config:
        os.environ["PAD_RUNTIME_CONFIG_PATH"] = str(args.runtime_config.expanduser().resolve())
    sys.path.insert(0, str(ROOT))
    import config
    from antispoof.predictor import AntiSpoofPredictor

    if not config.PAD_RUNTIME_CONFIG_PATH:
        raise ValueError("Cần runtime JSON để giữ đúng model contract; truyền --runtime-config")
    config_path = Path(config.PAD_RUNTIME_CONFIG_PATH)
    config_bytes = config_path.read_bytes()
    source_runtime = json.loads(config_bytes)
    predictor = AntiSpoofPredictor()
    print(f"Model: {predictor.model_path}\nConfig: {config_path}")
    print("Input: ảnh đã crop; real iff score >= threshold. Fit pooled binary ACER = HTER.")
    print("Calibration:")
    rows = score_images(calibration_dir, calibration_files, predictor, args.batch_size)
    scores, labels = row_arrays(rows)
    best, curve = select_threshold(scores, labels)
    original = metrics(scores, labels, predictor.logit_threshold)
    eval_rows, eval_result = None, None
    if evaluation:
        eval_dir, eval_files = evaluation
        print("Held-out evaluation (giữ nguyên threshold):")
        eval_rows = score_images(eval_dir, eval_files, predictor, args.batch_size)
        if {r["sha256"] for r in rows} & {r["sha256"] for r in eval_rows}:
            raise ValueError("Calibration/evaluation có ảnh trùng nội dung; cần tách tập trước khi chạy")
        eval_scores, eval_labels = row_arrays(eval_rows)
        eval_result = {
            "original": metrics(eval_scores, eval_labels, predictor.logit_threshold),
            "calibrated": metrics(eval_scores, eval_labels, best["threshold_logit"]),
        }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]
    output = args.output.expanduser().resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "dataset": str(calibration_dir), "fit_split": "local_domain_calibration",
        "unit": "image", "objective": "minimum pooled binary ACER/HTER",
        "decision_rule": "real iff score >= threshold", "tie_break": "lowest APCER, then largest threshold",
        "original_source_threshold": predictor.logit_threshold,
        "original_threshold_source": source_runtime.get("threshold_source"),
        "n_real": best["n_real"], "n_fake": best["n_fake"],
        "camera_temporal_validated": False, "report": str(output / "report.json"),
    }
    runtime_error = None
    try:
        runtime = export_runtime(source_runtime, predictor, best, provenance)
    except ValueError as exc:
        runtime_error = str(exc)
    report = {
        "schema": "pad_domain_threshold_calibration_v1", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": str(predictor.model_path),
        "model_sha256": hashlib.sha256(predictor.model_path.read_bytes()).hexdigest(),
        "source_runtime_config": str(config_path),
        "source_runtime_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_runtime": source_runtime,
        "effective_preprocessing": {"crop_mode": predictor.crop_mode,
            "model_img_size": predictor.model_img_size, "color_order": predictor.color_order,
            "apply_gamma": predictor.apply_gamma, "gamma_target": config.PAD_GAMMA_TARGET,
            "mean": predictor.mean, "std": predictor.std, "input_is_already_cropped": True},
        "providers": list(predictor.providers), "calibration": provenance,
        "original_metrics": original, "best_calibration_metrics": best,
        "threshold_candidates": len(curve),
        "evaluation_directory": str(evaluation[0]) if evaluation else None,
        "heldout_metrics": eval_result, "runtime_export_error": runtime_error,
        "metric_definition": "APCER=FA_fake/N_fake; BPCER=FR_real/N_real; pooled binary ACER=HTER=(APCER+BPCER)/2",
        "limitations": "Image-level fit metrics are optimistic; use subject/session-disjoint held-out data. No temporal voting evaluation. No per-attack worst-case ACER.",
    }
    for name, items in (("calibration_predictions.csv", rows), ("evaluation_predictions.csv", eval_rows)):
        if items:
            for row in items:
                row["predicted_label"] = "real" if row["score_logit"] >= best["threshold_logit"] else "fake"
                row["correct"] = row["predicted_label"] == row["label"]
            write_csv(output / name, items)
    write_csv(output / "threshold_sweep.csv", curve)
    write_json(output / "report.json", report)
    if runtime_error is None:
        write_json(output / "runtime_config_calibrated.json", runtime)
    print(f"\nThreshold logit: {best['threshold_logit']:.17g}")
    print(f"Threshold P(real): {best['threshold_probability']:.17g}")
    for name, result in (("Original / calibration", original), ("Best / calibration", best),
                         ("Calibrated / held-out", eval_result["calibrated"] if eval_result else None)):
        if result:
            print(f"{name}: APCER={result['APCER']:.4%}, BPCER={result['BPCER']:.4%}, "
                  f"ACER=HTER={result['ACER']:.4%}")
    print(f"Kết quả: {output}")
    if runtime_error:
        print(f"Không export runtime JSON: {runtime_error}")
    else:
        print(f"Để áp dụng, đổi PAD_RUNTIME_CONFIG_PATH trong face_auth/.env thành:\n{output / 'runtime_config_calibrated.json'}")
    print("Cấu hình hiện tại được giữ nguyên. Kết quả là theo ảnh, chưa kiểm chứng temporal voting.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1)
