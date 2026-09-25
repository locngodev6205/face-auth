"""
Real-time Face Anti-Spoofing & InsightFace SCRFD Face Detection Demo.
Integrated into SmartFace AIoT system.

Displays:
- Real vs Spoof decision
- Detailed Spoof Category: Physical (Print/Photo/Mask) vs Digital (Screen/Replay)
- Real-time Probability Breakdown (Real %, Print %, Screen %)
- Logit score vs Calibrated Threshold

Usage (from SmartFace root or face_auth directory):
    python face_auth/scripts/demo_antispoof.py
    python face_auth/scripts/demo_antispoof.py --image path/to/image.jpg
    python face_auth/scripts/demo_antispoof.py --liveness-model mnv3_e1_preliminary_v5_1_best.onnx
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add face_auth directory to python path
_HERE = Path(__file__).resolve().parent
_FACE_AUTH = _HERE.parent
_SMARTFACE_ROOT = _FACE_AUTH.parent
for p in [str(_HERE), str(_FACE_AUTH), str(_SMARTFACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from alignment.aligner import get_input_face
from antispoof import (
    AntiSpoofPredictor,
    crop,
    get_cpu_info,
    get_execution_provider_name,
    get_gpu_info,
)
from detection.detector import FaceDetector

# ── Color Definitions (BGR) ──────────────────────────────────────────────────
COLOR_REAL = (0, 220, 60)          # Vibrant Green for Real
COLOR_SPOOF_PRINT = (0, 0, 240)    # Red for Physical Spoof (Print/Paper/Photo)
COLOR_SPOOF_SCREEN = (230, 40, 210)# Magenta/Purple for Digital Spoof (Screen/Replay)
COLOR_SPOOF_GENERIC = (0, 0, 240)  # Red fallback
COLOR_LANDMARK = (0, 240, 255)     # Yellow
COLOR_CYAN = (255, 230, 0)         # Cyan
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)
COLOR_DARK_BG = (22, 22, 22)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_spoof_color_and_text(result: Dict) -> Tuple[Tuple[int, int, int], str, str]:
    """
    Determine box color, main title, and subtitle based on detailed class prediction.
    Returns: (color, main_label, category_description)
    """
    if result.get("is_real", False):
        return COLOR_REAL, "REAL", "Live Person"

    detailed = result.get("detailed_status", "").upper()
    if "PRINT" in detailed or "PHYSICAL" in detailed:
        return COLOR_SPOOF_PRINT, "SPOOF: PRINT", "Physical Attack (Photo/Paper)"
    elif "SCREEN" in detailed or "DIGITAL" in detailed:
        return COLOR_SPOOF_SCREEN, "SPOOF: SCREEN", "Digital Attack (Screen/Replay)"
    else:
        return COLOR_SPOOF_GENERIC, "SPOOF", "Presentation Attack"


def draw_probability_card(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    result: Dict,
    show_details: bool = True,
):
    """
    Draw a sleek diagnostic HUD card displaying probabilities and scores.
    Positioned smartly beside or below the face bounding box without clipping.
    """
    if not show_details or not result:
        return

    x1, y1, x2, y2 = [int(v) for v in bbox]
    frame_h, frame_w = frame.shape[:2]

    color, main_label, sub_desc = get_spoof_color_and_text(result)
    class_probs = result.get("class_probs", {})
    pad_score = result.get("pad_score", result.get("logit_diff", 0.0))
    threshold_logit = result.get("threshold_logit", -0.6826)

    # Prepare lines to render
    card_w = 230
    card_h = 130 if len(class_probs) >= 3 else 100

    # Smart positioning: try right, then left, then below
    if x2 + card_w + 10 <= frame_w:
        card_x = x2 + 10
        card_y = max(10, min(y1, frame_h - card_h - 10))
    elif x1 - card_w - 10 >= 0:
        card_x = x1 - card_w - 10
        card_y = max(10, min(y1, frame_h - card_h - 10))
    else:
        card_x = max(10, min(x1, frame_w - card_w - 10))
        card_y = min(y2 + 10, frame_h - card_h - 10)

    # Draw semi-transparent background overlay
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        COLOR_DARK_BG,
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Draw card border with classification color
    cv2.rectangle(
        frame,
        (card_x, card_y),
        (card_x + card_w, card_y + card_h),
        color,
        1,
        cv2.LINE_AA,
    )

    # Card title
    cv2.putText(
        frame,
        f"{main_label}",
        (card_x + 8, card_y + 20),
        FONT,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )

    # Sub description
    cv2.putText(
        frame,
        sub_desc,
        (card_x + 8, card_y + 36),
        FONT,
        0.36,
        COLOR_GRAY,
        1,
        cv2.LINE_AA,
    )

    # Draw probability bars
    bar_y = card_y + 54
    bar_w = 75
    bar_h = 7

    # Extract probabilities
    prob_items = []
    if "real" in class_probs:
        prob_items.append(("Real", class_probs.get("real", 0.0), COLOR_REAL))
    if "physical_spoof" in class_probs:
        prob_items.append(("Print", class_probs.get("physical_spoof", 0.0), COLOR_SPOOF_PRINT))
    if "screen_spoof" in class_probs:
        prob_items.append(("Screen", class_probs.get("screen_spoof", 0.0), COLOR_SPOOF_SCREEN))
    if not prob_items and "spoof" in class_probs:
        prob_items.append(("Real", class_probs.get("real", 0.0), COLOR_REAL))
        prob_items.append(("Spoof", class_probs.get("spoof", 0.0), COLOR_SPOOF_GENERIC))

    for label, prob, bar_color in prob_items:
        # Label & % text
        cv2.putText(
            frame,
            f"{label:<6}: {prob * 100:4.1f}%",
            (card_x + 8, bar_y + 6),
            FONT,
            0.38,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        # Bar background
        bx = card_x + card_w - bar_w - 10
        cv2.rectangle(
            frame,
            (bx, bar_y),
            (bx + bar_w, bar_y + bar_h),
            (50, 50, 50),
            cv2.FILLED,
        )

        # Bar fill
        fill_w = int(max(0.0, min(1.0, prob)) * bar_w)
        if fill_w > 0:
            cv2.rectangle(
                frame,
                (bx, bar_y),
                (bx + fill_w, bar_y + bar_h),
                bar_color,
                cv2.FILLED,
            )

        bar_y += 18

    # Score vs Threshold footer
    footer_text = f"d = {pad_score:+.2f} (thresh {threshold_logit:.2f})"
    cv2.putText(
        frame,
        footer_text,
        (card_x + 8, card_y + card_h - 8),
        FONT,
        0.36,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )


def draw_info_overlay(
    display_frame: np.ndarray,
    fps_history: List[float],
    cpu_info: str,
    gpu_info: str,
    provider_name: str,
    model_name: str,
    threshold: float,
    threshold_logit: float,
):
    """Draw system info and model configuration overlay."""
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

    info_y = 22
    line_height = 18
    font_scale = 0.44
    thickness = 1

    # Semi-transparent overlay box in top-left
    box_w = 340
    box_h = 135
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, display_frame, 0.28, 0, display_frame)

    cv2.putText(
        display_frame,
        f"SmartFace PAD Diagnostic Demo — FPS: {avg_fps:.1f}",
        (10, info_y),
        FONT,
        0.48,
        COLOR_CYAN,
        1,
        cv2.LINE_AA,
    )
    info_y += line_height

    cv2.putText(
        display_frame,
        f"Model: {model_name}",
        (10, info_y),
        FONT,
        font_scale,
        COLOR_WHITE,
        thickness,
        cv2.LINE_AA,
    )
    info_y += line_height

    cv2.putText(
        display_frame,
        f"Threshold: P(REAL) >= {threshold:.4f} (d >= {threshold_logit:.4f})",
        (10, info_y),
        FONT,
        font_scale,
        COLOR_WHITE,
        thickness,
        cv2.LINE_AA,
    )
    info_y += line_height

    # CPU/GPU & Provider
    cpu_short = cpu_info[:38] + "..." if len(cpu_info) > 38 else cpu_info
    cv2.putText(
        display_frame,
        f"CPU: {cpu_short}",
        (10, info_y),
        FONT,
        font_scale,
        COLOR_GRAY,
        thickness,
        cv2.LINE_AA,
    )
    info_y += line_height

    cv2.putText(
        display_frame,
        f"Device: {gpu_info if gpu_info else 'CPU'} ({provider_name})",
        (10, info_y),
        FONT,
        font_scale,
        COLOR_GRAY,
        thickness,
        cv2.LINE_AA,
    )
    info_y += line_height

    cv2.putText(
        display_frame,
        "Keys: 'c' toggle cards | 'i' toggle info | 'q' quit",
        (10, info_y),
        FONT,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )


def get_face_crop(
    frame_bgr: np.ndarray,
    face: dict,
    predictor: AntiSpoofPredictor,
    force_align: bool = False,
) -> np.ndarray:
    """
    Extract facial crop according to E1 contract:
    - Default: 1.55x expanded square crop around center with BORDER_REFLECT_101.
    - Optional force_align: ArcFace 5-point affine transformation (for comparison only).
    """
    bbox = face["bbox"]
    if force_align and face.get("landmarks"):
        crop_img = get_input_face(
            frame_bgr,
            bbox,
            face["landmarks"],
            output_size=(predictor.model_img_size, predictor.model_img_size),
        )
        if crop_img is not None:
            return crop_img

    return crop(frame_bgr, bbox, predictor.bbox_expansion_factor)


def process_camera(args, detector: FaceDetector, predictor: AntiSpoofPredictor):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    window_name = "SmartFace Anti-Spoofing Diagnostic Demo (SCRFD + MobileNetV3 PAD)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    show_info = True
    show_cards = args.card
    fps_history = []

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    provider_name = get_execution_provider_name(predictor.session)
    model_name = Path(predictor.model_path).name

    print("\nSmartFace Anti-Spoofing Diagnostic Demo started.")
    print("--------------------------------------------------")
    print(f"  Detector: SCRFD (buffalo_s, min_size={detector.min_face_size}px)")
    print(f"  PAD Model: {predictor.model_path}")
    print(f"  Input size: {predictor.model_img_size}x{predictor.model_img_size}")
    print(f"  Threshold: P(REAL)={predictor.threshold_probability:.4f} (d={predictor.logit_threshold:.4f})")
    print(f"  Crop factor: {predictor.bbox_expansion_factor:.2f}x (Affine align: {args.align})")
    print("--------------------------------------------------")
    print("Controls:")
    print("  'c' - Toggle probability cards")
    print("  'i' - Toggle system overlay")
    print("  'q' or ESC - Quit\n")

    while True:
        frame_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        # SCRFD detection on original BGR frame
        faces = detector.detect(frame)

        if faces:
            face_crops = []
            valid_faces = []
            for face in faces:
                try:
                    fc = get_face_crop(frame, face, predictor, force_align=args.align)
                    face_crops.append(fc)
                    valid_faces.append(face)
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Failed to crop face: {e}", file=sys.stderr)
                    continue

            if face_crops:
                results = predictor.predict_crops(face_crops)

                for face, result in zip(valid_faces, results):
                    color, main_label, sub_desc = get_spoof_color_and_text(result)
                    bbox = face["bbox"]

                    # Normalize bbox coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    if x2 < x1 or y2 < y1:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

                    # Draw Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw 5 Landmarks
                    if face.get("landmarks"):
                        for pt in face["landmarks"]:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, COLOR_LANDMARK, -1)

                    # Top label badge on bbox
                    tag = f"{main_label}"
                    (tw, th), base = cv2.getTextSize(tag, FONT, 0.55, 2)
                    label_y = max(y1 - 6, th + base + 4)
                    cv2.rectangle(frame, (x1, label_y - th - 4), (x1 + tw + 8, label_y + base), color, cv2.FILLED)
                    cv2.putText(frame, tag, (x1 + 4, label_y - 2), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                    # Draw Detailed Probability HUD Card
                    if show_cards:
                        draw_probability_card(frame, (x1, y1, x2, y2), result, show_details=True)

        # FPS calculation
        frame_time = time.perf_counter() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)

        # Overlay System Info
        if show_info:
            draw_info_overlay(
                frame,
                fps_history,
                cpu_info,
                gpu_info,
                provider_name,
                model_name,
                predictor.threshold_probability,
                predictor.logit_threshold,
            )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("i"):
            show_info = not show_info
        elif key == ord("c"):
            show_cards = not show_cards

    cap.release()
    cv2.destroyAllWindows()


def process_image(args, detector: FaceDetector, predictor: AntiSpoofPredictor):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from '{args.image}'", file=sys.stderr)
        return

    faces = detector.detect(image)
    if not faces:
        print("No faces detected in image.")
        return

    face_crops = []
    valid_faces = []
    for face in faces:
        try:
            fc = get_face_crop(image, face, predictor, force_align=args.align)
            face_crops.append(fc)
            valid_faces.append(face)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to crop face: {e}", file=sys.stderr)
            continue

    if face_crops:
        results = predictor.predict_crops(face_crops)
        for face, result in zip(valid_faces, results):
            color, main_label, sub_desc = get_spoof_color_and_text(result)
            bbox = face["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x2 < x1 or y2 < y1:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if face.get("landmarks"):
                for pt in face["landmarks"]:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 2, COLOR_LANDMARK, -1)

            tag = f"{main_label}"
            (tw, th), base = cv2.getTextSize(tag, FONT, 0.55, 2)
            label_y = max(y1 - 6, th + base + 4)
            cv2.rectangle(image, (x1, label_y - th - 4), (x1 + tw + 8, label_y + base), color, cv2.FILLED)
            cv2.putText(image, tag, (x1 + 4, label_y - 2), FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            draw_probability_card(image, (x1, y1, x2, y2), result, show_details=args.card)

    cv2.imshow("SmartFace PAD Result", image)
    print("Press any key to close window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="SmartFace Anti-Spoofing Diagnostic Demo (InsightFace SCRFD + MobileNetV3 3-Class PAD)"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to image file (if omitted, opens camera)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help=f"Camera index (default: {config.CAMERA_INDEX})")
    parser.add_argument("--threshold", type=float, default=config.PAD_THRESHOLD, help=f"Probability threshold P(REAL) (default: {config.PAD_THRESHOLD})")
    parser.add_argument("--threshold-logit", type=float, default=config.PAD_THRESHOLD_LOGIT, help=f"Logit threshold d=real-logsumexp(spoof) (default: {config.PAD_THRESHOLD_LOGIT})")
    parser.add_argument("--detector-model", type=str, default=config.MODEL_PACK_NAME, help=f"Detector model pack (default: {config.MODEL_PACK_NAME})")
    parser.add_argument("--min-face-size", type=int, default=config.DETECTOR_MIN_FACE_SIZE, help=f"Minimum face size in px (default: {config.DETECTOR_MIN_FACE_SIZE})")
    parser.add_argument("--liveness-model", type=str, default=config.PAD_MODEL_FILENAME, help=f"PAD model filename or full path (default: {config.PAD_MODEL_FILENAME})")
    parser.add_argument("--bbox-factor", type=float, default=config.PAD_BBOX_EXPANSION_FACTOR, help=f"Crop expansion factor (default: {config.PAD_BBOX_EXPANSION_FACTOR})")
    parser.add_argument("--align", action="store_true", default=False, help="Force ArcFace 5-point affine alignment instead of standard expanded crop (default: False)")
    parser.add_argument("--card", action="store_true", default=True, help="Display detailed probability HUD card (default: True)")
    parser.add_argument("--no-card", dest="card", action="store_false", help="Hide probability HUD card")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Resolve PAD model path
    pad_model_path = Path(args.liveness_model).expanduser()
    if not pad_model_path.is_absolute():
        pad_model_path = _FACE_AUTH / "antispoof" / "models" / pad_model_path

    if not pad_model_path.exists():
        print(f"[ERROR] PAD model file not found: {pad_model_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize SCRFD Detector with min_face_size
    detector = FaceDetector(
        model_name=args.detector_model,
        conf_thresh=config.DETECTOR_CONF_THRESH,
        min_face_size=args.min_face_size,
    )

    # Initialize AntiSpoofPredictor
    effective_logit = args.threshold_logit
    if effective_logit is None and args.threshold == config.PAD_THRESHOLD:
        effective_logit = config.PAD_THRESHOLD_LOGIT

    predictor = AntiSpoofPredictor(
        model_path=str(pad_model_path.resolve()),
        threshold=args.threshold,
        threshold_logit=effective_logit,
        bbox_expansion_factor=args.bbox_factor,
        apply_gamma=config.PAD_GAMMA_ENABLED,
        color_order=config.PAD_COLOR_ORDER,
    )

    if args.image is None:
        process_camera(args, detector, predictor)
    else:
        process_image(args, detector, predictor)


if __name__ == "__main__":
    main()
