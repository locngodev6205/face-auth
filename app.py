"""
Face Authentication App — SmartFace AIoT.

Pipeline mỗi frame:
    Camera → Detection (SCRFD/YunNet) → Tracking (IOU) → [PAD liveness check] → Alignment → Embedding → DB Search → Display

CLI usage:
    python app.py
    python app.py --no-pad           # tắt anti-spoofing
    python app.py --no-fps           # tắt FPS overlay
    python app.py --pad-model mnv4_best_224.onnx   # chọn model PAD khác
    python app.py --no-pad --no-fps  # chạy nhanh nhất
"""

import argparse
import math
import sys
import time
import cv2
import config 
from pathlib import Path
from collections import deque
from recognition.embedder import FaceEmbedder
from alignment.aligner import get_input_face
from database.vector_db import VectorDB, decide_identity
from tracking.tracker import FaceTracker

# ── Chọn detector theo APP_DETECTOR trong config / .env ────────────────────
if config.APP_DETECTOR == "scrfd":
    from detection.detector import FaceDetector
    _DETECTOR_BACKEND = "scrfd"
else:
    from detection.yunnet_detector import FaceDetector
    _DETECTOR_BACKEND = "yunnet"


# ── Màu sắc cho các trạng thái track ────────────────────────────────────────
COLOR_PENDING   = (180, 180, 180)   # Xám    — chưa nhận diện lần nào
COLOR_LOCKED    = (0, 210, 80)      # Xanh   — đã nhận diện & đang locked
COLOR_REVERIFY  = (0, 165, 255)     # Cam    — đang re-verify
COLOR_UNKNOWN   = (0, 40, 220)      # Đỏ     — UNKNOWN

# Màu cho trạng thái PAD
COLOR_REAL      = (0, 210, 80)      # Xanh   — mặt thật
COLOR_SPOOF     = (0, 0, 220)       # Đỏ     — giả mạo


# ── Hàm vẽ ──────────────────────────────────────────────────────────────────

def draw_track(frame, bbox, name, score, is_pending, is_reverifying, is_spoof=False, is_pad_pending=False):
    """
    Vẽ bounding box và label lên frame theo trạng thái track.

    Trạng thái:
        - SPOOF (đỏ)         : PAD phát hiện giả mạo — ưu tiên cao nhất.
        - PAD Pending (xám)  : đang tích lũy vote PAD, chưa cho phép nhận diện.
        - Pending (xám)      : track vừa tạo, chưa nhận diện.
        - Locked (xanh)      : đã nhận diện, đang trong thời gian locked.
        - Re-verify (cam)    : chờ re-verify, giữ tên cũ.
        - Unknown (đỏ đậm)   : không khớp ai trong DB.
    """
    x1, y1, x2, y2 = bbox

    if is_spoof:
        color = COLOR_SPOOF
        label = "⚠ SPOOF"
    elif is_pad_pending:
        color = COLOR_PENDING
        label = "Kiem tra PAD..."
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


def orchestrate_track_step(
    track,
    pad_enabled_rt: bool,
    app_mode: str,
    db,
    embedder=None,
    frame=None,
    detection=None,
    recognize_interval_seconds: float = config.RECOGNIZE_INTERVAL_SECONDS,
    match_threshold: float = config.MATCH_THRESHOLD,
    model_version: str = config.EMBEDDING_MODEL_VERSION,
    attendance_gap_minutes: int = config.ATTENDANCE_GAP_MINUTES,
    attendance_stable_count: int = config.ATTENDANCE_STABLE_COUNT,
) -> dict:
    """
    Điều phối nhận diện và điểm danh cho một track:
        1. Kiểm tra trạng thái PAD (spoof, pending, real, hoặc disabled).
        2. Quyết định có chạy ArcFace recognition & DB search hay không.
        3. Cập nhật kết quả nhận diện và reset recognition stability nếu spoof/pending.
        4. Quyết định và thực thi ghi nhận điểm danh (attendance logging).
    """
    is_spoof = pad_enabled_rt and track.is_spoof
    is_pad_pending = pad_enabled_rt and (not track.pad_ready)

    if is_spoof or is_pad_pending:
        track.stable_recognitions = 0

    should_recognize = track.needs_recognition(recognize_interval_seconds, pad_enabled=pad_enabled_rt)
    recognized = False
    recognition_called = False

    if should_recognize and embedder is not None and frame is not None and detection is not None:
        recognition_called = True
        bbox = detection["bbox"]
        landmarks = detection.get("landmarks")
        try:
            aligned_face = get_input_face(frame, bbox, landmarks, embedder.input_size)
            embedding = (
                embedder.embed_aligned(aligned_face)
                if aligned_face is not None
                else None
            )
            if embedding is not None:
                rows = db.search(embedding, top_k=5, model_version=model_version)
                employee_id, name, score = decide_identity(rows, threshold=match_threshold)
                track.update_result(employee_id, name, score)
                recognized = True
        except Exception as e:
            print(f"[Pipeline error] {e}")

    attendance_attempted = False
    attendance_success = None
    attendance_reason = None
    last_ts = None

    if (
        app_mode != "NONE"
        and not is_spoof
        and not is_pad_pending
        and not track.is_pending
        and track.employee_id
        and track.can_log_attendance(attendance_gap_minutes)
        and track.stable_recognitions >= attendance_stable_count
    ):
        attendance_attempted = True
        success, reason, last_ts = db.log_attendance(
            employee_id=track.employee_id,
            action=app_mode,
            gap_minutes=attendance_gap_minutes,
            face_similarity=float(track.score) if track.score is not None else None,
            liveness_score=track.last_pad_score,
        )
        attendance_success = success
        attendance_reason = reason
        # Ba trạng thái phân biệt:
        #   success=True              → ghi log thành công, set local cooldown
        #   success=False, last_ts≠None → DB báo đã có attendance trong cooldown,
        #                               sync local cooldown theo timestamp DB để
        #                               tránh retry spam mỗi chu kỳ frame
        #   success=False, last_ts=None → lỗi DB thực sự, không tạo cooldown giả
        if success:
            track.last_attendance_time = (
                last_ts.timestamp() if last_ts else time.time()
            )
        elif last_ts is not None:
            track.last_attendance_time = last_ts.timestamp()

    return {
        "should_recognize": should_recognize,
        "is_spoof": is_spoof,
        "is_pad_pending": is_pad_pending,
        "recognized": recognized,
        "recognition_called": recognition_called,
        "attendance_attempted": attendance_attempted,
        "attendance_success": attendance_success,
        "attendance_reason": attendance_reason,
        "last_ts": last_ts,
    }


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
        "--pad-model", type=str, default=config.PAD_MODEL_PATH,
        help=f"Đường dẫn model PAD (mặc định: {config.PAD_MODEL_PATH})"
    )
    parser.add_argument(
        "--pad-threshold", type=float, default=config.PAD_THRESHOLD,
        help=f"Ngưỡng xác suất P(REAL) cho PAD (mặc định: {config.PAD_THRESHOLD})"
    )
    parser.add_argument(
        "--pad-threshold-logit", type=float, default=None,
        help="Ghi đè trực tiếp ngưỡng d=real-logsumexp(spoof), tránh nhầm đơn vị"
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
    # App Mode (Attendance)
    parser.add_argument(
        "--mode", type=str, default=config.ATTENDANCE_MODE, choices=["checkin", "checkout", "none"],
        help=f"Chế độ ứng dụng: checkin, checkout hoặc none (mặc định: {config.ATTENDANCE_MODE})"
    )
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    effective_pad_threshold_logit = args.pad_threshold_logit
    if effective_pad_threshold_logit is None and args.pad_threshold == config.PAD_THRESHOLD:
        effective_pad_threshold_logit = config.PAD_THRESHOLD_LOGIT

    # ── LOAD ONCE ──────────────────────────────────────────────────────────────
    if _DETECTOR_BACKEND == "scrfd":
        detector = FaceDetector(
            model_name=config.MODEL_PACK_NAME,
            ctx_id=config.MODEL_CTX_ID,
            det_size=config.DETECTOR_DET_SIZE,
            conf_thresh=config.DETECTOR_CONF_THRESH,
            min_face_size=config.DETECTOR_MIN_FACE_SIZE,
        )
    else:  # yunnet
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
        pad_min_votes=config.PAD_MIN_VOTES,
        pad_stale_timeout=config.PAD_STALE_TIMEOUT_SECONDS,
    )

    # ── Khởi tạo PAD (nếu bật) ───────────────────────────────────────────────
    pad_predictor = None
    if args.pad_enabled:
        try:
            from antispoof import AntiSpoofPredictor
            _pad_model_path = Path(args.pad_model).expanduser()
            if not _pad_model_path.is_absolute():
                _pad_model_path = Path(__file__).resolve().parent / "antispoof" / "models" / _pad_model_path
            pad_predictor = AntiSpoofPredictor(
                model_path=str(_pad_model_path.resolve()),
                threshold=args.pad_threshold,
                threshold_logit=effective_pad_threshold_logit,
                model_img_size=config.PAD_MODEL_IMG_SIZE,
                bbox_expansion_factor=config.PAD_BBOX_EXPANSION_FACTOR,
                mean=config.PAD_MEAN,
                std=config.PAD_STD,
                apply_gamma=config.PAD_GAMMA_ENABLED,
                color_order=config.PAD_COLOR_ORDER,
            )
            print(f"  PAD              = ON  ({pad_predictor.model_path})")
            print(f"  PAD_BBOX_EXPAND  = {pad_predictor.bbox_expansion_factor:.2f}x")
            print(f"  PAD threshold    = p={pad_predictor.threshold_probability:.12f}, d={pad_predictor.logit_threshold:.12f} ({pad_predictor.threshold_input_type})")
            print(f"  PAD_GAMMA        = {'ON' if config.PAD_GAMMA_ENABLED else 'OFF'}  (target luma={config.PAD_GAMMA_TARGET:.0f})")
            print(f"  PAD providers    = {list(pad_predictor.providers)}")
            print(f"  PAD input/output = {pad_predictor.input_metadata.name} {pad_predictor.input_metadata.shape} -> {pad_predictor.output_metadata.name} {pad_predictor.output_metadata.shape}")
        except Exception as e:
            print(f"[ERROR] Không thể tải model PAD: {e}.", file=sys.stderr)
            print("[ERROR] PAD đang được yêu cầu (--pad hoặc PAD_ENABLED=true). Hệ thống dừng khởi động để đảm bảo an toàn.", file=sys.stderr)
            print("[ERROR] Nếu muốn chạy hệ thống không có kiểm tra liveness PAD, hãy khởi động với cờ --no-pad.", file=sys.stderr)
            db.close()
            sys.exit(1)
    else:
        print("  PAD              = OFF")

    # ── Camera ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        db.close()
        raise RuntimeError(f"Không thể mở camera index {args.camera}.")

    # Giữ input camera ổn định trên các backend khác nhau (AVFoundation/MSMF/V4L2).
    # Một số backend không hỗ trợ BUFFERSIZE hoặc có thể làm tròn resolution; các giá trị
    # thực tế được in ra ngay sau đây để chẩn đoán thay vì giả định cap.set() đã thành công.
    if config.CAMERA_BUFFER_SIZE > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    try:
        camera_backend = cap.getBackendName()
    except (AttributeError, cv2.error):
        camera_backend = "UNKNOWN"
    actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_buffer = cap.get(cv2.CAP_PROP_BUFFERSIZE)

    print("Starting face authentication system...")
    print(
        f"  CAMERA          = backend={camera_backend} "
        f"requested={config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}@{config.CAMERA_FPS} "
        f"actual={actual_width}x{actual_height}@{actual_fps:.2f} "
        f"buffer={actual_buffer:.2f}"
    )
    print(f"  DETECTOR         = {_DETECTOR_BACKEND.upper()}")
    print(f"  MATCH_THRESHOLD  = {config.MATCH_THRESHOLD}")
    print(f"  RECOGNIZE_EVERY  = {config.RECOGNIZE_INTERVAL_SECONDS}s")
    print(f"  PAD_EVERY        = {config.PAD_INTERVAL_SECONDS}s")
    print(f"  DETECT_EVERY     = {config.DETECTION_INTERVAL_SECONDS}s (0 = every frame)")
    print(f"  MIN_FACE_SIZE    = {config.DETECTOR_MIN_FACE_SIZE}px")
    print(f"  APP_MODE         = {args.mode.upper()}")
    print(f"  FPS overlay      = {'ON' if args.show_fps else 'OFF'}")
    print("Controls: 'q' quit | 'p' toggle PAD | 'f' toggle FPS")

    pad_enabled_rt = (pad_predictor is not None)  # runtime toggle
    show_fps_rt    = args.show_fps                 # runtime toggle
    app_mode       = args.mode.upper()
    attendance_msg = ""
    attendance_msg_color = (0, 255, 0)
    attendance_msg_time = 0

    frame_time_deque: deque = deque(maxlen=max(1, config.FPS_AVG_WINDOW))
    avg_fps = 0.0
    frame_id = 0
    last_detection_time = None
    counters = {
        "total_frames": 0,
        "detector_calls": 0,
        "tracker_updates": 0,
        "pad_calls": 0,
        "recognition_calls": 0,
        "detector_ms": 0.0,
        "tracker_ms": 0.0,
        "pad_ms": 0.0,
        "recognition_ms": 0.0,
    }

    def print_runtime_summary():
        total = max(counters["total_frames"], 1)
        print("Runtime summary:")
        print(
            f"  frames={counters['total_frames']} "
            f"detector_calls={counters['detector_calls']} "
            f"({counters['detector_calls'] / total:.1%} of frames) "
            f"tracker_updates={counters['tracker_updates']} "
            f"pad_calls={counters['pad_calls']} "
            f"recognition_calls={counters['recognition_calls']}"
        )
        for name in ("detector", "tracker", "pad", "recognition"):
            calls = counters[f"{name}_calls"] if name != "tracker" else counters["tracker_updates"]
            mean_ms = counters[f"{name}_ms"] / max(calls, 1)
            print(f"  {name}_mean_ms={mean_ms:.2f}")

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("Failed to read from camera.")
                break
            frame_id += 1
            counters["total_frames"] += 1

            # ── Detection cadence + bbox propagation ─────────────────────────
            now = time.monotonic()
            detector_due = (
                last_detection_time is None
                or config.DETECTION_INTERVAL_SECONDS <= 0
                or (now - last_detection_time) >= config.DETECTION_INTERVAL_SECONDS
                or tracker.needs_redetection()
            )
            detection_called = False
            if detector_due:
                detector_t0 = time.perf_counter()
                try:
                    detections = detector.detect(frame)
                except Exception as e:
                    print(f"[Detector error] {e}")
                    detections = []
                counters["detector_ms"] += (time.perf_counter() - detector_t0) * 1000.0
                counters["detector_calls"] += 1
                last_detection_time = now
                detection_called = True
                tracker_t0 = time.perf_counter()
                tracked = tracker.update(detections, frame=frame, detector_called=True)
            else:
                tracker_t0 = time.perf_counter()
                tracked = tracker.update([], frame=frame, detector_called=False)
            counters["tracker_ms"] += (time.perf_counter() - tracker_t0) * 1000.0
            counters["tracker_updates"] += 1

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
                    pad_t0 = time.perf_counter()
                    try:
                        pad_batch = pad_predictor.predict_crops(face_crops)
                        counters["pad_calls"] += len(pad_batch)
                        # Đẩy verdict vào rolling window của từng track
                        if pad_batch:
                            tid_to_track = {t.track_id: t for _, t in tracked}
                            for tid, res in zip(track_ids, pad_batch):
                                if tid in tid_to_track and "is_real" in res:
                                    pad_track = tid_to_track[tid]
                                    pad_track.update_pad(
                                        is_real=res["is_real"],
                                        pad_score=res.get("pad_score", res.get("logit_diff", None)),
                                    )
                                    if config.PAD_DIAGNOSTIC_LOG:
                                        score = float(res.get("pad_score", res.get("logit_diff", 0.0)))
                                        p_real = 1.0 / (1.0 + math.exp(-score))
                                        print(
                                            f"[PAD] frame_id={frame_id} timestamp={time.time():.6f} "
                                            f"track_id={tid} detector_called={detection_called} "
                                            f"bbox_source={detection.get('bbox_source', 'UNKNOWN')} "
                                            f"bbox={pad_track.bbox} face_width={pad_track.bbox[2]-pad_track.bbox[0]} "
                                            f"face_height={pad_track.bbox[3]-pad_track.bbox[1]} "
                                            f"logits=({res.get('real_logit')},{res.get('spoof_logit')}) "
                                            f"d={score:.6f} p_real={p_real:.6f} "
                                            f"threshold_logit={res.get('threshold_logit', pad_predictor.logit_threshold):.6f} "
                                            f"raw={'REAL' if res['is_real'] else 'SPOOF'} "
                                            f"smoothed={pad_track.pad_status}"
                                        )
                    except Exception as e:
                        print(f"[PAD error] {e}")
                    finally:
                        counters["pad_ms"] += (time.perf_counter() - pad_t0) * 1000.0

            # ── Recognition + Attendance + Display ───────────────────────────
            for detection, track in tracked:
                bbox = detection["bbox"]
                recognition_t0 = time.perf_counter()

                orch_res = orchestrate_track_step(
                    track=track,
                    detection=detection,
                    frame=frame,
                    embedder=embedder,
                    db=db,
                    pad_enabled_rt=pad_enabled_rt,
                    app_mode=app_mode,
                    recognize_interval_seconds=config.RECOGNIZE_INTERVAL_SECONDS,
                    match_threshold=config.MATCH_THRESHOLD,
                    model_version=config.EMBEDDING_MODEL_VERSION,
                    attendance_gap_minutes=config.ATTENDANCE_GAP_MINUTES,
                    attendance_stable_count=config.ATTENDANCE_STABLE_COUNT,
                )
                counters["recognition_ms"] += (time.perf_counter() - recognition_t0) * 1000.0
                if orch_res["recognition_called"]:
                    counters["recognition_calls"] += 1

                # Trạng thái hiển thị
                name, score = track.name, track.score
                is_pending = track.is_pending
                is_reverifying = (
                    not is_pending
                    and not orch_res["is_pad_pending"]
                    and not orch_res["is_spoof"]
                    and track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS, pad_enabled=pad_enabled_rt)
                )

                if orch_res["attendance_attempted"]:
                    attendance_msg_time = time.time()
                    last_ts = orch_res["last_ts"]
                    if orch_res["attendance_success"]:
                        attendance_msg = f"{name}: {app_mode} SUCCESS"
                        attendance_msg_color = (0, 255, 0)
                        ts_str = f" lúc {last_ts.astimezone().strftime('%H:%M:%S')}" if last_ts else ""
                        print(f"[ATTENDANCE] {attendance_msg}{ts_str}")
                    else:
                        reason = orch_res["attendance_reason"]
                        attendance_msg = f"{name}: {reason}"
                        attendance_msg_color = (0, 165, 255)
                        ts_str = f" (Gần nhất: {last_ts.astimezone().strftime('%H:%M:%S')})" if last_ts else ""
                        print(f"[ATTENDANCE BLOCKED] {name}: {reason}{ts_str}")

                draw_track(
                    frame, bbox, name, score,
                    is_pending=is_pending,
                    is_reverifying=is_reverifying,
                    is_spoof=orch_res["is_spoof"],
                    is_pad_pending=orch_res["is_pad_pending"],
                )


            # ── Overlay FPS & PAD badge ───────────────────────────────────────
            if show_fps_rt:
                draw_fps(frame, avg_fps)
            draw_pad_badge(frame, pad_enabled_rt)
            
            # ── Draw mode & attendance message ────────────────────────────────
            cv2.putText(frame, f"MODE: {app_mode}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
            if attendance_msg and (time.time() - attendance_msg_time < 3.0):
                # Hiển thị thông báo (thành công hoặc cảnh báo) trong 3 giây
                cv2.putText(frame, attendance_msg, (10, frame.shape[0] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, attendance_msg_color, 2, cv2.LINE_AA)

            cv2.imshow("SmartFace — Face Auth", frame)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            # Đo toàn bộ vòng lặp, bao gồm capture, inference, imshow và waitKey.
            # Dùng tổng thời gian / số frame thay vì mean(1/dt), tránh FPS bị thổi phồng
            # khi thời gian từng frame dao động giữa các backend camera.
            t1 = time.perf_counter()
            frame_time_deque.append(max(t1 - t0, 1e-6))
            avg_fps = len(frame_time_deque) / sum(frame_time_deque)

            if key == ord("q") or key == 27:
                break
            elif key == ord("p"):
                if pad_predictor is None:
                    try:
                        from antispoof import AntiSpoofPredictor
                        _pad_model_path = Path(args.pad_model).expanduser()
                        if not _pad_model_path.is_absolute():
                            _pad_model_path = Path(__file__).resolve().parent / "antispoof" / "models" / _pad_model_path
                        pad_predictor = AntiSpoofPredictor(
                            model_path=str(_pad_model_path.resolve()),
                            threshold=args.pad_threshold,
                            threshold_logit=effective_pad_threshold_logit,
                            model_img_size=config.PAD_MODEL_IMG_SIZE,
                            bbox_expansion_factor=config.PAD_BBOX_EXPANSION_FACTOR,
                            mean=config.PAD_MEAN,
                            std=config.PAD_STD,
                            apply_gamma=config.PAD_GAMMA_ENABLED,
                            color_order=config.PAD_COLOR_ORDER,
                        )
                        print(f"[PAD] Model loaded on demand: {args.pad_model}")
                    except Exception as e:
                        print(f"[PAD error] Không thể tải model PAD: {e}")

                if pad_predictor is not None:
                    pad_enabled_rt = not pad_enabled_rt
                    if pad_enabled_rt:
                        for t in tracker.tracks:
                            t.reset_pad()
                    print(f"[PAD] {'ON' if pad_enabled_rt else 'OFF'}")
                else:
                    print("[PAD] Không thể bật — model PAD chưa sẵn sàng.")
            elif key == ord("f"):
                show_fps_rt = not show_fps_rt

    finally:
        print_runtime_summary()
        cap.release()
        cv2.destroyAllWindows()
        db.close()


if __name__ == "__main__":
    main()
