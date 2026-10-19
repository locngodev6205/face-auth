"""Thu ảnh PAD chưa gán nhãn: click SAVE / Space / S; Q / Esc để thoát.

Chạy từ repo: python3 face_auth/scripts/capture_domain_faces.py --subject-id p01
Ảnh PNG giữ nguyên pixel camera, không vẽ bbox, gamma, align hay resize.
Crop dùng hình học minifasnet_train_v1 (R7), mặc định mở rộng 1.5x.
Để chấm R7: resize crop trực tiếp về 80x80 và preprocess; KHÔNG crop 1.5x lần nữa.
Hoặc chạy pipeline trên frame.png và bbox gốc trong metadata.json.
Gán real/fake riêng cho từng face trong metadata.json sau khi thu thập.
Tách người/phiên giữa tập hiệu chỉnh và tập đánh giá threshold.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
from uuid import uuid4

import cv2

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "PAD domain capture"


def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid4().hex[:8]


def expanded_bounds(frame_shape, bbox, scale=1.5):
    """Native-resolution bounds, identical to runtime minifasnet_train_v1.

    x2/y2 are inclusive, as in antispoof/preprocess.py. Near an edge the
    rectangle shifts inside the frame; scale is limited by frame dimensions.
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    values = (x1, y1, x2, y2, scale)
    if not all(math.isfinite(v) for v in values) or scale < 1:
        raise ValueError("BBox phải hữu hạn; scale phải >= 1")
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0 or width < 2 or height < 2:
        raise ValueError("Kích thước frame/bbox không hợp lệ")
    used_scale = min((height - 1) / box_h, (width - 1) / box_w, scale)
    new_w, new_h = box_w * used_scale, box_h * used_scale
    center_x, center_y = box_w / 2 + x1, box_h / 2 + y1
    left, top = center_x - new_w / 2, center_y - new_h / 2
    right, bottom = center_x + new_w / 2, center_y + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width - 1:
        left -= right - width + 1
        right = width - 1
    if bottom > height - 1:
        top -= bottom - height + 1
        bottom = height - 1
    bounds = tuple(map(int, (left, top, right, bottom)))
    lx, ty, rx, by = bounds
    if not (0 <= lx <= rx < width and 0 <= ty <= by < height):
        raise ValueError(f"Crop vượt frame: {bounds}")
    return bounds, used_scale


def write_png(path, image):
    # imencode + Python file I/O supports Unicode output paths on Windows.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Không mã hóa được ảnh: {path}")
    path.write_bytes(encoded.tobytes())


def save_capture(session_dir, frame, detections, scale, captured_at):
    """Publish a complete capture only after every image/metadata write succeeds."""
    if not detections:
        return None
    capture_id = unique_id()
    target = session_dir / capture_id
    staging = Path(tempfile.mkdtemp(prefix=".pending_", dir=session_dir))
    try:
        faces = []
        for index, detection in enumerate(detections, 1):
            bounds, used_scale = expanded_bounds(frame.shape, detection["bbox"], scale)
            left, top, right, bottom = bounds
            patch = frame[top:bottom + 1, left:right + 1]
            name = f"face_{index:02d}.png"
            write_png(staging / name, patch)
            faces.append({
                "file": name, "label": None, "subject_id": None,
                "bbox_xyxy": list(map(int, detection["bbox"])),
                "detector_score": float(detection["score"]),
                "landmarks": detection.get("landmarks"),
                "crop_xyxy_inclusive": list(bounds),
                "effective_scale": used_scale,
                "image_size_wh": [patch.shape[1], patch.shape[0]],
            })
        write_png(staging / "frame.png", frame)
        metadata = {
            "schema": "pad_domain_capture_v1", "capture_id": capture_id,
            "captured_at_utc": captured_at, "frame_file": "frame.png",
            "frame_size_wh": [frame.shape[1], frame.shape[0]],
            "crop_mode": "minifasnet_train_v1_native",
            "requested_scale": scale, "faces": faces,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging)
        raise
    return target


def create_detector(config):
    if config.APP_DETECTOR == "scrfd":
        from detection.detector import FaceDetector
        return FaceDetector(
            model_name=config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID,
            det_size=config.DETECTOR_DET_SIZE,
            conf_thresh=config.DETECTOR_CONF_THRESH,
            min_face_size=config.DETECTOR_MIN_FACE_SIZE,
        )
    if config.APP_DETECTOR == "yunnet":
        from detection.yunnet_detector import FaceDetector
        return FaceDetector(
            model_path=config.DETECTOR_MODEL_PATH,
            conf_thresh=config.DETECTOR_CONF_THRESH,
            nms_thresh=config.DETECTOR_NMS_THRESH, top_k=config.DETECTOR_TOP_K,
            min_face_size=config.DETECTOR_MIN_FACE_SIZE, margin=config.DETECTOR_MARGIN,
        )
    raise ValueError(f"APP_DETECTOR không hỗ trợ: {config.APP_DETECTOR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, help="Mặc định CAMERA_INDEX trong .env")
    parser.add_argument("--output", type=Path, default=ROOT / "data/domain_capture")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--subject-id", default="", help="Mã người của phiên; nhiều người: gán riêng trong metadata")
    parser.add_argument("--notes", default="", help="Camera, ánh sáng, thiết bị tấn công, điều kiện thu...")
    args = parser.parse_args()
    if not math.isfinite(args.scale) or args.scale < 1:
        parser.error("--scale phải hữu hạn và >= 1")

    sys.path.insert(0, str(ROOT))
    import config

    detector = create_detector(config)
    camera = config.CAMERA_INDEX if args.camera is None else args.camera
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được camera {camera}")
        if config.CAMERA_BUFFER_SIZE > 0:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
        if config.CAMERA_FORCE_RESOLUTION:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

        session_dir = args.output.expanduser().resolve() / unique_id()
        session_dir.mkdir(parents=True, exist_ok=False)
        session = {
            "schema": "pad_domain_session_v1", "session_id": session_dir.name,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "subject_id": args.subject_id or None, "notes": args.notes,
            "camera_index": camera, "camera_backend": cap.getBackendName(),
            "detector": config.APP_DETECTOR, "model_pack": config.MODEL_PACK_NAME,
            "detector_model_path": config.DETECTOR_MODEL_PATH,
            "detector_confidence": config.DETECTOR_CONF_THRESH,
            "min_face_size": config.DETECTOR_MIN_FACE_SIZE,
            "detector_det_size": list(config.DETECTOR_DET_SIZE),
            "detector_margin": config.DETECTOR_MARGIN,
            "requested_scale": args.scale,
            "runtime_config_reference": config.PAD_RUNTIME_CONFIG_PATH,
            "label_values": ["real", "fake"],
        }
        (session_dir / "session.json").write_text(
            json.dumps(session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Lưu dữ liệu tại: {session_dir}")
        print("Click SAVE / Space / S: lưu TẤT CẢ mặt. Q / Esc: thoát.")
        print("Nhãn nằm trong metadata.json: label=null -> real/fake; subject_id riêng nếu có nhiều người.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        state = {"save": False, "button": (0, 0, 0, 0)}

        def on_mouse(event, x, y, flags, param):
            left, top, right, bottom = state["button"]
            if event == cv2.EVENT_LBUTTONDOWN and left <= x <= right and top <= y <= bottom:
                state["save"] = True

        cv2.setMouseCallback(WINDOW, on_mouse)
        total = 0
        status = "Ready"
        status_until = 0.0
        while True:
            ok, frame = cap.read()
            captured_at = datetime.now(timezone.utc).isoformat()
            if not ok or frame is None:
                raise RuntimeError("Không đọc được frame camera; dữ liệu đã lưu vẫn được giữ.")
            # Detect every displayed frame: saved boxes always belong to this frame.
            detections = detector.detect(frame)
            height, width = frame.shape[:2]
            preview = cv2.copyMakeBorder(frame, 0, 105, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for index, detection in enumerate(detections, 1):
                x1, y1, x2, y2 = detection["bbox"]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), 2)
                bounds, _ = expanded_bounds(frame.shape, detection["bbox"], args.scale)
                lx, ty, rx, by = bounds
                cv2.rectangle(preview, (lx, ty), (rx, by), (0, 200, 255), 1)
                cv2.putText(preview, f"#{index}", (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            state["button"] = (10, height + 10, min(150, width - 10), height + 48)
            left, top, right, bottom = state["button"]
            cv2.rectangle(preview, (left, top), (right, bottom), (50, 130, 50), -1)
            cv2.putText(preview, "SAVE", (left + 12, top + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(preview, f"Faces: {len(detections)} | Saved: {total} | Space/S: save | Q: quit",
                        (10, height + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if time.monotonic() < status_until:
                cv2.putText(preview, status, (10, height + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if state["save"] or key in (32, ord("s"), ord("S")):
                state["save"] = False
                try:
                    path = save_capture(session_dir, frame, detections, args.scale, captured_at)
                    if path is None:
                        status = "No faces - nothing saved"
                    else:
                        total += len(detections)
                        status = f"Saved {len(detections)} face(s)"
                        print(f"{status}: {path}")
                except (OSError, ValueError, cv2.error) as exc:
                    status = "SAVE FAILED - see terminal"
                    print(f"Lưu thất bại: {exc}", file=sys.stderr)
                status_until = time.monotonic() + 3
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        raise SystemExit(1)
