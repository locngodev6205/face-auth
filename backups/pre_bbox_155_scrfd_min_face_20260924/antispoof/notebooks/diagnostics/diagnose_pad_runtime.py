#!/usr/bin/env python3
"""Deterministic single-image PAD runtime diagnostic.

This intentionally has no tracker or temporal smoothing. It reports the exact
crop, tensor contract, ONNX artifact, logits, binary score and decision for a
small matrix of preprocessing/threshold variants.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Allow ``python antispoof/notebooks/diagnostics/diagnose_pad_runtime.py`` from
# the repository root without requiring an installed package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from antispoof.predictor import AntiSpoofPredictor
from antispoof.preprocess import crop, preprocess_batch


E1_MEAN = [0.5931, 0.4690, 0.4229]
E1_STD = [0.2471, 0.2214, 0.2157]
E1_THRESHOLD_LOGIT = -0.682607114315033


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_bbox(image: np.ndarray, explicit_bbox: Iterable[int] | None):
    if explicit_bbox is not None:
        return tuple(int(v) for v in explicit_bbox), "CLI_BBOX"

    cascade = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(boxes) == 0:
        raise RuntimeError(
            "Không tìm thấy bbox. Truyền --bbox x1 y1 x2 y2 để diagnostic deterministic."
        )
    x, y, w, h = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    return (int(x), int(y), int(x + w), int(y + h)), "HAAR_STATIC_DIAGNOSTIC"


def run_variant(
    label: str,
    model_path: Path,
    face_crop: np.ndarray,
    *,
    apply_gamma: bool,
    color_order: str,
    mean,
    std,
    threshold_logit: float | None = None,
    threshold: float | None = None,
):
    predictor = AntiSpoofPredictor(
        model_path=str(model_path),
        threshold=threshold,
        threshold_logit=threshold_logit,
        apply_gamma=apply_gamma,
        color_order=color_order,
        mean=mean,
        std=std,
    )
    tensor = preprocess_batch(
        [face_crop],
        predictor.model_img_size,
        mean=predictor.mean,
        std=predictor.std,
        apply_gamma=predictor.apply_gamma,
        convert_rgb=predictor.convert_rgb,
    )
    logits = predictor.session.run([], {predictor.input_name: tensor})[0][0]
    result = predictor.process_with_logits(logits)
    print(f"\n[{label}]")
    print(f"  gamma_enabled={apply_gamma} color_conversion=BGR->RGB:{predictor.convert_rgb}")
    print(f"  mean={predictor.mean} std={predictor.std}")
    print(
        f"  tensor shape={tensor.shape} dtype={tensor.dtype} "
        f"min={tensor.min():.6f} max={tensor.max():.6f} mean={tensor.mean():.6f}"
    )
    print(
        f"  logits={[float(x) for x in logits]} d={result['pad_score']:.9f} "
        f"p_real={1.0 / (1.0 + math.exp(-result['pad_score'])):.9f} "
        f"threshold_logit={result['threshold_logit']:.9f} "
        f"threshold_probability={result['threshold_probability']:.9f} "
        f"decision={'REAL' if result['is_real'] else 'SPOOF'}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(config.PAD_MODEL_PATH),
        help="ONNX model; default is the effective config model",
    )
    parser.add_argument("--bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--save-crop", type=Path, default=None)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Không đọc được image: {args.image}")
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    bbox, bbox_source = locate_bbox(image, args.bbox)
    face_crop = crop(image, bbox, bbox_expansion_factor=1.5)
    data_path = Path(f"{model_path}.data")

    print("# PAD runtime diagnostic")
    print(f"image={args.image.resolve()} shape={image.shape} dtype={image.dtype}")
    print(f"model={model_path}")
    print(f"model_sha256={sha256(model_path)}")
    if data_path.exists():
        print(f"external_data={data_path} sha256={sha256(data_path)}")
    else:
        print("external_data=<not present>")
    print(f"bbox_source={bbox_source} raw_bbox={bbox}")
    print(f"expanded_bbox_factor=1.5 crop_shape={face_crop.shape}")
    if args.save_crop is not None:
        args.save_crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_crop), face_crop)
        print(f"saved_crop={args.save_crop.resolve()}")

    probe = AntiSpoofPredictor(
        model_path=str(model_path),
        threshold_logit=E1_THRESHOLD_LOGIT,
        apply_gamma=False,
        color_order="RGB",
        mean=E1_MEAN,
        std=E1_STD,
    )
    print(f"providers={probe.providers}")
    print(
        f"input name={probe.input_metadata.name} shape={probe.input_metadata.shape} "
        f"type={probe.input_metadata.type}"
    )
    print(
        f"output name={probe.output_metadata.name} shape={probe.output_metadata.shape} "
        f"type={probe.output_metadata.type}"
    )

    run_variant(
        "A CORRECT E1",
        model_path,
        face_crop,
        apply_gamma=False,
        color_order="RGB",
        mean=E1_MEAN,
        std=E1_STD,
        threshold_logit=E1_THRESHOLD_LOGIT,
    )
    run_variant(
        "B GAMMA ON",
        model_path,
        face_crop,
        apply_gamma=True,
        color_order="RGB",
        mean=E1_MEAN,
        std=E1_STD,
        threshold_logit=E1_THRESHOLD_LOGIT,
    )
    run_variant(
        "C WRONG BGR",
        model_path,
        face_crop,
        apply_gamma=False,
        color_order="BGR",
        mean=E1_MEAN,
        std=E1_STD,
        threshold_logit=E1_THRESHOLD_LOGIT,
    )
    run_variant(
        "D THRESHOLD 0.5",
        model_path,
        face_crop,
        apply_gamma=False,
        color_order="RGB",
        mean=E1_MEAN,
        std=E1_STD,
        threshold=0.5,
    )
    run_variant(
        "E LEGACY NORMALIZATION (no mean/std)",
        model_path,
        face_crop,
        apply_gamma=False,
        color_order="RGB",
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        threshold_logit=E1_THRESHOLD_LOGIT,
    )


if __name__ == "__main__":
    main()
