"""
Real-time Face Anti-Spoofing & YunNet Face Detection Demo.
Integrated into SmartFace AIoT system.

Usage (from SmartFace root or face_auth directory):
    python face_auth/scripts/demo_antispoof_yunnet.py
    python face_auth/scripts/demo_antispoof_yunnet.py --image path/to/image.jpg

    python face_auth/scripts/demo_antispoof_yunnet.py \
  --liveness-model face_auth/antispoof/models/mnv4_best_224.onnx

"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add face_auth directory to python path
_HERE = Path(__file__).resolve().parent
_FACE_AUTH = _HERE.parent
_SMARTFACE_ROOT = _FACE_AUTH.parent
for p in [str(_HERE), str(_FACE_AUTH), str(_SMARTFACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from detection.yunnet_detector import YunNetFaceDetector
from alignment.aligner import get_input_face
from antispoof import (
    AntiSpoofPredictor,
    crop,
    get_cpu_info,
    get_gpu_info,
    get_execution_provider_name,
)

# Color definitions (BGR)
COLOR_REAL = (0, 255, 0)      # Green
COLOR_SPOOF = (0, 0, 255)     # Red
COLOR_LANDMARK = (0, 255, 255) # Yellow
COLOR_CYAN = (255, 255, 0)    # Cyan
COLOR_WHITE = (255, 255, 255) # White
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_info_overlay(display_frame, fps_history, cpu_info, gpu_info, provider_name):
    """Draw system info and FPS overlay on frame."""
    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

    info_y = 25
    line_height = 20
    font_scale = 0.5
    thickness = 1

    cv2.putText(
        display_frame,
        f"FPS: {avg_fps:.1f}",
        (5, info_y),
        FONT,
        font_scale,
        COLOR_CYAN,
        thickness,
    )
    info_y += line_height

    cpu_lines = []
    max_chars_per_line = 55
    words = cpu_info.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_chars_per_line:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                cpu_lines.append(current_line)
            current_line = word
    if current_line:
        cpu_lines.append(current_line)

    for i, cpu_line in enumerate(cpu_lines[:2]):
        cv2.putText(
            display_frame,
            f"CPU: {cpu_line}" if i == 0 else cpu_line,
            (5, info_y),
            FONT,
            font_scale,
            COLOR_WHITE,
            thickness,
        )
        info_y += line_height

    if gpu_info:
        cv2.putText(
            display_frame,
            f"GPU: {gpu_info}",
            (5, info_y),
            FONT,
            font_scale,
            COLOR_WHITE,
            thickness,
        )
        info_y += line_height
    else:
        cv2.putText(
            display_frame,
            "GPU: No GPU detected",
            (5, info_y),
            FONT,
            font_scale,
            COLOR_WHITE,
            thickness,
        )
        info_y += line_height

    cv2.putText(
        display_frame,
        f"Provider: {provider_name}",
        (5, info_y),
        FONT,
        font_scale,
        COLOR_WHITE,
        thickness,
    )
    info_y += line_height
    cv2.putText(
        display_frame,
        "Press 'i' to toggle info | 'q' to quit",
        (5, info_y),
        FONT,
        0.4,
        (200, 200, 200),
        1,
    )


def get_face_crop(image_rgb: np.ndarray, face: dict, predictor: AntiSpoofPredictor, use_align: bool = True) -> np.ndarray:
    """
    Cắt và chuẩn hóa khuôn mặt sử dụng hàm get_input_face trong alignment.aligner:
    - Nếu use_align=True và có 5 điểm landmarks: xoay và căn mặt chuẩn theo ArcFace template.
    - Nếu không: fallback crop theo bbox.
    """
    bbox = face["bbox"]
    landmarks = face.get("landmarks") if use_align else None

    crop_img = get_input_face(
        image_rgb,
        bbox,
        landmarks,
        output_size=(predictor.model_img_size, predictor.model_img_size),
    )
    if crop_img is not None:
        return crop_img

    # Fallback nếu get_input_face trả về None
    return crop(image_rgb, bbox, predictor.bbox_expansion_factor)


def process_camera(args, detector: YunNetFaceDetector, predictor: AntiSpoofPredictor):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {args.camera}", file=sys.stderr)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "SmartFace Anti-Spoofing Demo (YunNet + AntiSpoof ONNX)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    show_info = True
    fps_history = []

    cpu_info = get_cpu_info()
    gpu_info = get_gpu_info()
    provider_name = get_execution_provider_name(predictor.session)

    print("Controls:")
    print("  'q' - Quit")
    print("  'i' - Toggle info display")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect(frame_rgb, margin=args.margin)  # detect frame

        if faces:
            face_crops = []
            valid_faces = []
            for face in faces:
                try:
                    face_crop = get_face_crop(
                        frame_rgb, face, predictor, use_align=args.align
                    )
                    face_crops.append(face_crop)
                    valid_faces.append(face)
                except Exception as e:
                    if args.verbose:
                        print(f"Warning: Failed to crop/align face: {e}", file=sys.stderr)
                    continue

            if face_crops:
                results = predictor.predict_crops(face_crops)

                for face, result in zip(valid_faces, results):
                    color = COLOR_REAL if result["is_real"] else COLOR_SPOOF
                    bbox = face["bbox"]
                    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Draw 5 landmarks
                    if "landmarks" in face:
                        for pt in face["landmarks"]:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, COLOR_LANDMARK, -1)

                    align_str = " (Aligned)" if args.align and face.get("landmarks") else ""
                    label = f"{result.get('detailed_status', result['status'].upper())}{align_str}: {result['logit_diff']:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x, max(0, y - 10)),
                        FONT,
                        0.6,
                        color,
                        2,
                    )

        frame_time = time.time() - frame_start
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)

        display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

        if show_info:
            draw_info_overlay(
                display_frame, fps_history, cpu_info, gpu_info, provider_name
            )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("i"):
            show_info = not show_info

    cap.release()
    cv2.destroyAllWindows()


def process_image(args, detector: YunNetFaceDetector, predictor: AntiSpoofPredictor):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image from '{args.image}'", file=sys.stderr)
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect(image_rgb, margin=args.margin)

    if not faces:
        print("No faces detected in image.")
        return

    face_crops = []
    valid_faces = []
    for face in faces:
        try:
            face_crop = get_face_crop(
                image_rgb, face, predictor, use_align=args.align
            )
            face_crops.append(face_crop)
            valid_faces.append(face)
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to crop/align face: {e}", file=sys.stderr)
            continue

    if face_crops:
        results = predictor.predict_crops(face_crops)

        for face, result in zip(valid_faces, results):
            color = COLOR_REAL if result["is_real"] else COLOR_SPOOF
            bbox = face["bbox"]
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            if "landmarks" in face:
                for pt in face["landmarks"]:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, COLOR_LANDMARK, -1)

            align_str = " (Aligned)" if args.align and face.get("landmarks") else ""
            label = f"{result.get('detailed_status', result['status'].upper())}{align_str}: {result['logit_diff']:.2f}"
            cv2.putText(image, label, (x, max(0, y - 10)), FONT, 0.6, color, 2)

    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # định nghĩa tham số dòng lệnh cho chương trình
    parser = argparse.ArgumentParser(description="SmartFace Anti-Spoofing Demo (YunNet + AntiSpoof ONNX)")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (if omitted, opens camera)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")   # mặc định camera của máy 
    parser.add_argument("--threshold", type=float, default=0.5, help="Real/Spoof threshold (default: 0.5)") # ngưỡng phân loại
    parser.add_argument("--margin", type=int, default=5, help="Face edge margin (default: 5)")  # biên mở rộng
    parser.add_argument("--detector-model", type=str, default=None, help="Path to YunNet ONNX detector model")  # đường dẫn tới model detect
    parser.add_argument("--liveness-model", type=str, default=None, help="Path to AntiSpoof ONNX model")    # đường dẫn tới model antiproof
    parser.add_argument("--align", action="store_true", default=True, help="Enable face alignment using 5 landmarks (default: True)")
    parser.add_argument("--no-align", action="store_false", dest="align", help="Disable face alignment, fallback to 1.5x bbox crop")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose error logging")  # true: log lỗi chi tiết

    # đọc tham số lưu vào args
    args = parser.parse_args()

    detector = YunNetFaceDetector(model_path=args.detector_model)
    predictor = AntiSpoofPredictor(model_path=args.liveness_model, threshold=args.threshold)

    if args.image is None:
        process_camera(args, detector, predictor)
    else:
        process_image(args, detector, predictor)


if __name__ == "__main__":
    main()
