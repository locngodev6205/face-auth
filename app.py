"""
Face Authentication App — SmartFace AIoT.

Pipeline mỗi frame:
    Camera → Detection (YunNet) → Tracking (IOU) → [PAD liveness check] → Alignment → Embedding → DB Search → Display

CLI usage:
    python app.py
    python app.py --no-pad           # tắt anti-spoofing
    python app.py --no-fps           # tắt FPS overlay
    python app.py --pad-model mnv4_best_224.onnx   # chọn model PAD khác
    python app.py --no-pad --no-fps  # chạy nhanh nhất
"""

import argparse
import time
import cv2
import config
from pathlib import Path
from collections import deque
from detection.yunnet_detector import FaceDetector
from recognition.embedder import FaceEmbedder
from alignment.aligner import get_input_face
from database.vector_db import VectorDB, decide_identity
from tracking.tracker import FaceTracker


# ── Màu sắc cho các trạng thái track ────────────────────────────────────────
COLOR_PENDING   = (180, 180, 180)   # Xám    — chưa nhận diện lần nào
COLOR_LOCKED    = (0, 210, 80)      # Xanh   — đã nhận diện & đang locked
COLOR_REVERIFY  = (0, 165, 255)     # Cam    — đang re-verify
COLOR_UNKNOWN   = (0, 40, 220)      # Đỏ     — UNKNOWN

# Màu cho trạng thái PAD
COLOR_REAL      = (0, 210, 80)      # Xanh   — mặt thật
COLOR_SPOOF     = (0, 0, 220)       # Đỏ     — giả mạo


# ── Hàm vẽ ──────────────────────────────────────────────────────────────────

def draw_track(frame, bbox, name, score, is_pending, is_reverifying, is_spoof=False):
    """
    Vẽ bounding box và label lên frame theo trạng thái track.

    Trạng thái:
        - SPOOF (đỏ)      : PAD phát hiện giả mạo — ưu tiên cao nhất.
        - Pending (xám)   : track vừa tạo, chưa nhận diện.
        - Locked (xanh)   : đã nhận diện, đang trong thời gian locked.
        - Re-verify (cam) : chờ re-verify, giữ tên cũ.
        - Unknown (đỏ đậm): không khớp ai trong DB.
    """
    x1, y1, x2, y2 = bbox

    if is_spoof:
        color = COLOR_SPOOF
        label = "⚠ SPOOF"
    elif is_pending:
        color = COLOR_PENDING
        label = "Nhan dien..."
    elif name == "UNKNOWN":
        color = COLOR_UNKNOWN
        label = f"UNKNOWN  {score:.2f}"
    elif is_reverifying:
        color = COLOR_REVERIFY
        label = f"{name}  {score:.2f}"
    else:
        color = COLOR_LOCKED
        label = f"[OK] {name}  {score:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    font_thick = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y1 = max(y1 - text_h - baseline - 4, 0)
    label_y2 = max(y1 - 2, text_h + baseline + 4)
    cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 6, label_y2), color, cv2.FILLED)
    cv2.putText(frame, label, (x1 + 3, label_y2 - baseline - 2),
                font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    """Vẽ FPS ở góc trên-trái của frame."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def draw_pad_badge(frame, enabled: bool):
    """Vẽ badge nhỏ cho biết module PAD đang bật hay tắt."""
    text  = "PAD: ON" if enabled else "PAD: OFF"
    color = (0, 200, 80) if enabled else (100, 100, 100)
    cv2.putText(frame, text, (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


# ── Hàm parse CLI ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SmartFace — Face Authentication with optional PAD (Liveness Detection)"
    )
    # PAD
    pad_group = parser.add_mutually_exclusive_group()
    pad_group.add_argument(
        "--pad", dest="pad_enabled", action="store_true", default=config.PAD_ENABLED,
        help=f"Bật module PAD anti-spoofing (mặc định: {'ON' if config.PAD_ENABLED else 'OFF'})"
    )
    pad_group.add_argument(
        "--no-pad", dest="pad_enabled", action="store_false",
        help="Tắt module PAD anti-spoofing"
    )
    parser.add_argument(
        "--pad-model", type=str, default=config.PAD_MODEL_FILENAME,
        help=f"Tên file model PAD trong antispoof/models/ (mặc định: {config.PAD_MODEL_FILENAME})"
    )
    parser.add_argument(
        "--pad-threshold", type=float, default=config.PAD_THRESHOLD,
        help=f"Ngưỡng Real/Spoof cho PAD (mặc định: {config.PAD_THRESHOLD})"
    )
    # FPS overlay
    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument(
        "--fps", dest="show_fps", action="store_true", default=config.SHOW_FPS,
        help=f"Hiển thị FPS overlay (mặc định: {'ON' if config.SHOW_FPS else 'OFF'})"
    )
    fps_group.add_argument(
        "--no-fps", dest="show_fps", action="store_false",
        help="Tắt hiển thị FPS"
    )
    # Camera
    parser.add_argument(
        "--camera", type=int, default=config.CAMERA_INDEX,
        help=f"Chỉ số camera (mặc định: {config.CAMERA_INDEX})"
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── LOAD ONCE ────────────────────────────────────────────────────────────
    detector = FaceDetector(
        model_path=config.DETECTOR_MODEL_PATH,
        conf_thresh=config.DETECTOR_CONF_THRESH,
        nms_thresh=config.DETECTOR_NMS_THRESH,
        top_k=config.DETECTOR_TOP_K,
        min_face_size=config.DETECTOR_MIN_FACE_SIZE,
        margin=config.DETECTOR_MARGIN,
    )
    embedder = FaceEmbedder(config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID)
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
    )
    tracker = FaceTracker(
        recognize_interval_seconds=config.RECOGNIZE_INTERVAL_SECONDS,
        iou_threshold=config.TRACK_IOU_THRESHOLD,
        max_missing_frames=config.TRACK_MAX_MISSING_FRAMES,
        pad_smooth_window=config.PAD_SMOOTH_WINDOW,
        pad_spoof_min_ratio=config.PAD_SPOOF_MIN_RATIO,
        pad_interval_seconds=config.PAD_INTERVAL_SECONDS,
    )

    # ── Khởi tạo PAD (nếu bật) ───────────────────────────────────────────────
    pad_predictor = None
    if args.pad_enabled:
        try:
            from antispoof import AntiSpoofPredictor
            _pad_model_path = str(
                Path(__file__).parent / "antispoof" / "models" / args.pad_model
            )
            pad_predictor = AntiSpoofPredictor(
                model_path=_pad_model_path,
                threshold=args.pad_threshold,
            )
            print(f"  PAD              = ON  ({args.pad_model}, threshold={args.pad_threshold})")
        except Exception as e:
            print(f"[Warning] Không thể tải model PAD: {e}. Tiếp tục không có PAD.")
            pad_predictor = None
    else:
        print("  PAD              = OFF")

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.CAMERA_FPS)

    print("Starting face authentication system...")
    print(f"  MATCH_THRESHOLD  = {config.MATCH_THRESHOLD}")
    print(f"  RECOGNIZE_EVERY  = {config.RECOGNIZE_INTERVAL_SECONDS}s")
    print(f"  PAD_EVERY        = {config.PAD_INTERVAL_SECONDS}s")
    print(f"  MIN_FACE_SIZE    = {config.DETECTOR_MIN_FACE_SIZE}px")
    print(f"  FPS overlay      = {'ON' if args.show_fps else 'OFF'}")
    print("Controls: 'q' quit | 'p' toggle PAD | 'f' toggle FPS")

    pad_enabled_rt = (pad_predictor is not None)  # runtime toggle
    show_fps_rt    = args.show_fps                 # runtime toggle

    fps_deque: deque = deque(maxlen=config.FPS_AVG_WINDOW)
    avg_fps = 0.0

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break

            # ── Detection ────────────────────────────────────────────────────
            try:
                detections = detector.detect(frame)
            except Exception as e:
                print(f"[Detector error] {e}")
                detections = []

            tracked = tracker.update(detections)

            # ── PAD batch — chỉ chạy cho các track đến hạn PAD_INTERVAL_SECONDS ─────
            if pad_enabled_rt and pad_predictor is not None and tracked:
                face_crops = []
                track_ids  = []
                for detection, track in tracked:
                    # Bỏ qua track chưa đến hạn chạy PAD lại
                    if not track.needs_pad(config.PAD_INTERVAL_SECONDS):
                        continue
                    bbox = detection["bbox"]
                    try:
                        from antispoof.preprocess import crop as pad_crop
                        fc = pad_crop(frame, bbox, pad_predictor.bbox_expansion_factor)
                        face_crops.append(fc)
                        track_ids.append(track.track_id)
                    except Exception:
                        pass
                if face_crops:
                    try:
                        pad_batch = pad_predictor.predict_crops(face_crops)
                        # Đẩy verdict vào rolling window của từng track
                        tid_to_track = {t.track_id: t for _, t in tracked}
                        for tid, res in zip(track_ids, pad_batch):
                            if tid in tid_to_track:
                                tid_to_track[tid].update_pad(res.get("is_real", True))
                    except Exception as e:
                        print(f"[PAD error] {e}")

            # ── Recognition + Display ─────────────────────────────────────────
            for detection, track in tracked:
                bbox      = detection["bbox"]
                landmarks = detection.get("landmarks")

                # Lấy trạng thái PAD đã smoothed từ rolling window của track
                is_spoof = pad_enabled_rt and track.is_spoof

                # Chỉ chạy recognition khi mặt không phải SPOOF (hoặc PAD tắt)
                if not is_spoof and track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS):
                    try:
                        aligned_face = get_input_face(
                            frame, bbox, landmarks, embedder.input_size
                        )
                        embedding = (
                            embedder.embed_aligned(aligned_face)
                            if aligned_face is not None
                            else None
                        )
                        if embedding is not None:
                            rows = db.search(embedding, top_k=5)
                            name, score = decide_identity(rows, threshold=config.MATCH_THRESHOLD)
                            track.update_result(name, score)
                    except Exception as e:
                        print(f"[Pipeline error] {e}")

                # Xác định trạng thái
                name, score = track.name, track.score
                is_pending     = track.is_pending
                is_reverifying = (
                    not is_pending
                    and track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS)
                )

                draw_track(frame, bbox, name, score, is_pending, is_reverifying, is_spoof=is_spoof)

            # ── Overlay FPS & PAD badge ───────────────────────────────────────
            if show_fps_rt:
                draw_fps(frame, avg_fps)
            draw_pad_badge(frame, pad_enabled_rt)

            cv2.imshow("SmartFace — Face Auth", frame)

            # ── FPS tính sau khi imshow để bao gồm cả render time ─────────────
            t1 = time.perf_counter()
            fps_deque.append(1.0 / max(t1 - t0, 1e-6))
            avg_fps = sum(fps_deque) / len(fps_deque)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("p"):
                if pad_predictor is not None:
                    pad_enabled_rt = not pad_enabled_rt
                    print(f"[PAD] {'ON' if pad_enabled_rt else 'OFF'}")
                else:
                    print("[PAD] Không thể bật — model PAD chưa được tải thành công.")
            elif key == ord("f"):
                show_fps_rt = not show_fps_rt

    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()