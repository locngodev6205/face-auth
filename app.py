"""
Face Authentication App — SmartFace AIoT.

Camera thread: Detection → Tracking → PAD → Alignment snapshot → Display.
Recognition worker: ArcFace → cloud DB / per-track centroid cache → result queue.

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
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import config
from pathlib import Path
from collections import deque
from recognition.embedder import FaceEmbedder
from recognition.worker import RecognitionWorker
from alignment.aligner import get_input_face
from database.vector_db import VectorDB, decide_identity
from tracking.tracker import FaceTracker
from vision_pipeline import _pad_face_skip_reason, update_track_pad, limit_faces

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
COLOR_INVALID   = (0, 165, 255)     # Cam    — đầu vào mặt không hợp lệ


def _opencv_display_text(text):
    """Convert Unicode text to ASCII for OpenCV's built-in Hershey font."""
    text = str(text).replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(
        char for char in normalized
        if not unicodedata.combining(char) and ord(char) < 128
    )


# ── Hàm vẽ ──────────────────────────────────────────────────────────────────

def draw_track(
    frame, bbox, name, score, is_pending, is_reverifying,
    is_spoof=False, is_pad_pending=False, pad_invalid_reason=None,
    recognition_error=False,
    recognition_status="",
    stable_label=False,
):
    """
    Vẽ bounding box và label lên frame theo trạng thái track.

    Trạng thái:
        - SPOOF (đỏ)         : PAD phát hiện giả mạo — ưu tiên cao nhất.
        - Invalid face (cam) : đầu vào chưa đủ điều kiện để chạy PAD.
        - PAD Pending (xám)  : đang tích lũy vote PAD, chưa cho phép nhận diện.
        - Pending (xám)      : track vừa tạo, chưa nhận diện.
        - Locked (xanh)      : đã nhận diện, đang trong thời gian locked.
        - Re-verify (xanh)   : giữ màu danh tính trước, ghi LAST và trạng thái chờ.
        - Unknown (đỏ đậm)   : không khớp ai trong DB.
    """
    x1, y1, x2, y2 = bbox
    # Keep operational waits/errors visible; hide recognition backend telemetry.
    recognition_status = " | ".join(
        part.strip() for part in recognition_status.split("|")
        if part.strip() and part.strip().split()[0].upper() not in ("CLOUD", "CACHE")
    )

    if is_spoof:
        color = COLOR_SPOOF
        label = "⚠ SPOOF"
    elif pad_invalid_reason:
        invalid_detail = {
            "low_detector_confidence": "LOW CONF",
            "face_too_close_to_frame_edge": "EDGE",
            "incomplete_landmarks": "LANDMARKS",
            "invalid_landmarks": "LANDMARKS",
            "landmark_outside_face": "LANDMARKS",
            "landmark_at_face_edge": "LANDMARKS",
        }.get(pad_invalid_reason, "INPUT")
        color = COLOR_INVALID
        label = f"INVALID FACE: {invalid_detail}"
    elif recognition_error:
        color = COLOR_INVALID
        label = "RECOGNITION ERROR"
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
        color = COLOR_LOCKED
        label = f"[LAST] {name}  {score:.2f}"
    else:
        color = COLOR_LOCKED
        label = f"[OK] {name}  {score:.2f}"

    detail = ""
    if stable_label and not (is_spoof or is_pad_pending or pad_invalid_reason or recognition_error):
        if not is_pending and name != "UNKNOWN":
            label = f"[ID] {name}"
            if is_reverifying:
                detail = "RECHECK"
        detail = " | ".join(part for part in (detail, recognition_status) if part)
        if is_pending and recognition_status.startswith(("WAIT", "QUEUE")):
            color = COLOR_REVERIFY
    elif recognition_status and not (is_spoof or is_pad_pending or pad_invalid_reason):
        label += f" | {recognition_status}"
        if recognition_status.startswith(("WAIT", "QUEUE")) and is_pending and not recognition_error:
            color = COLOR_REVERIFY

    label = _opencv_display_text(label)
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
    if detail:
        # Separate changing telemetry from the identity header; never shift its text.
        detail = _opencv_display_text(detail)
        cv2.putText(frame, detail, (x1 + 3, min(y2 + 20, frame.shape[0] - 5)),
                    font, 0.45, color, 1, cv2.LINE_AA)


def draw_pad_margin(frame, bbox, track, threshold):
    """Display the latest raw PAD margin separately from the temporal verdict."""
    if track.last_pad_threshold is not None:
        threshold = track.last_pad_threshold
    score = track.last_pad_score
    stale = (
        track.last_pad_time is not None
        and config.PAD_STALE_TIMEOUT_SECONDS > 0
        and time.monotonic() - track.last_pad_time > config.PAD_STALE_TIMEOUT_SECONDS
    )
    if track.pad_invalid_reason or score is None or stale or not math.isfinite(score):
        reason = "INVALID" if track.pad_invalid_reason else "STALE" if stale else "WAIT"
        lines = [f"PAD d: -- ({reason})", f"t={threshold:+.3f}   d-t: --"]
        color = COLOR_PENDING
    else:
        gap = score - threshold
        raw = "REAL" if gap >= 0 else "SPOOF"
        lines = [f"d={score:+.3f}   t={threshold:+.3f}",
                 f"d-t={gap:+.3f}   raw {raw}"]
        color = (110, 235, 150) if gap >= 0 else (110, 130, 255)

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.50
    width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 12
    height = 48
    x = max(0, min(int(bbox[0]), frame.shape[1] - width))
    y = max(0, min(int(bbox[3]) + 28, frame.shape[0] - height))
    cv2.rectangle(frame, (x, y), (min(x + width, frame.shape[1] - 1), y + height - 1),
                  (25, 25, 25), cv2.FILLED)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x + 5, y + 18 + index * 22), font, scale,
                    (235, 235, 235) if index == 0 else color, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    """Vẽ FPS ở góc trên-trái của frame."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def draw_pad_badge(frame, enabled: bool):
    """Vẽ badge nhỏ cho biết module PAD đang bật hay tắt."""
    text  = "PAD: ON | d=z_real-z_fake" if enabled else "PAD: OFF"
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
                employee_code, name, score = decide_identity(rows, threshold=match_threshold)
                track.update_result(employee_code, name, score)
                recognized = True
        except Exception as e:
            track.recognition_error = True
            print(
                f"[Recognition error] track_id={track.track_id} "
                f"stage=alignment_embedding_or_db type={type(e).__name__}: {e}"
            )
        if not recognized:
            track.invalidate_recognition(retry_delay=True)
            if recognition_called:
                track.recognition_error = True

    attendance_attempted = False
    attendance_success = None
    attendance_reason = None
    last_ts = None

    # Disabled in Embed-only scope: Python không còn ghi attendance_logs trực tiếp.
    # Backend Prisma là writer duy nhất cho bảng attendance_logs.
    # if (
    #     app_mode != "NONE"
    #     and not is_spoof
    #     and not is_pad_pending
    #     and not track.is_pending
    #     and track.employee_id
    #     and track.can_log_attendance(attendance_gap_minutes)
    #     and track.stable_recognitions >= attendance_stable_count
    # ):
    #     attendance_attempted = True
    #     success, reason, last_ts = db.log_attendance(
    #         employee_id=track.employee_id,
    #         action=app_mode,
    #         gap_minutes=attendance_gap_minutes,
    #         face_similarity=float(track.score) if track.score is not None else None,
    #         liveness_score=track.last_pad_score,
    #     )
    #     attendance_success = success
    #     attendance_reason = reason
    #     if success:
    #         track.last_attendance_time = (
    #             last_ts.timestamp() if last_ts else time.time()
    #         )
    #     elif last_ts is not None:
    #         track.last_attendance_time = last_ts.timestamp()

    return {
        "should_recognize": should_recognize,
        "is_spoof": is_spoof,
        "is_pad_pending": is_pad_pending,
        "pad_invalid_reason": (
            track.pad_invalid_reason if pad_enabled_rt else None
        ),
        "recognized": recognized,
        "recognition_called": recognition_called,
        "attendance_attempted": attendance_attempted,
        "attendance_success": attendance_success,
        "attendance_reason": attendance_reason,
        "last_ts": last_ts,
    }


def collect_recognition_results(worker, tracked, pad_enabled_rt):
    """Only the camera thread mutates tracks; late results need valid continuity."""
    valid = {}
    visible = {track.track_id: track for _, track in tracked}
    for track_id, track in {**worker.visible_tracks, **visible}.items():
        key = (track.track_id, track.verification_generation)
        eligible = track_id in visible and track.missing_frames == 0 and (
            not pad_enabled_rt or (track.is_real and not track.pad_invalid_reason)
        )
        if eligible:
            track.recognition_eligible = True
            valid[key] = track
        elif track.recognition_eligible or (
            key in worker.pending and not worker.pending[key].cancelled.is_set()
        ):
            track.verification_generation += 1
            track.invalidate_recognition()
            track.recognition_status = ""
            track.recognition_eligible = False
    worker.visible_tracks = visible
    worker.retain(set(valid))
    for result in worker.poll():
        track = valid.get(result.key)
        if track is None:
            continue
        track.recognition_latency = result.latency
        if result.error:
            track.invalidate_recognition(retry_delay=True)
            track.recognition_error = True
            track.recognition_status = f"ERROR {result.error} {result.latency:.1f}s"
        else:
            track.update_result(*result.identity)
            track.update_display_identity(
                result.identity, config.RECOGNITION_LABEL_CONFIRMATIONS,
                config.RECOGNITION_LABEL_UNKNOWN_CONFIRMATIONS,
            )
            track.recognition_status = f"{result.source} {result.latency:.1f}s"
        print(f"[Recognition] track_id={track.track_id} {track.recognition_status}")


def attendance_confirmation_ready(track):
    """Readiness only; the demo never writes attendance directly."""
    return bool(track.employee_id is not None and not track.recognition_error
                and track.stable_recognitions >= config.ATTENDANCE_STABLE_COUNT)


def submit_recognition_step(worker, track, detection, frame, embedder, pad_enabled_rt):
    is_spoof = pad_enabled_rt and track.is_spoof
    is_pad_pending = pad_enabled_rt and not track.pad_ready
    if is_spoof or is_pad_pending:
        track.stable_recognitions = 0
    key = (track.track_id, track.verification_generation)
    submitted = False
    if key in worker.pending and not worker.pending[key].cancelled.is_set():
        wait = time.monotonic() - worker.pending[key].submitted
        track.recognition_status = f"WAIT {wait:.1f}s"
    elif (track.missing_frames == 0 and
          track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS, pad_enabled=pad_enabled_rt)):
        try:
            # Small crop snapshot; no full frames retained in the work queue.
            face = get_input_face(frame, detection["bbox"], detection.get("landmarks"),
                                  embedder.input_size)
            if face is None:
                raise ValueError("Alignment failed")
            submitted = worker.submit(key, face)
            track.recognition_status = "WAIT 0.0s" if submitted else "QUEUE FULL"
            if submitted:
                track.recognition_error = False
        except Exception as exc:
            track.invalidate_recognition(retry_delay=True)
            track.recognition_error = True
            track.recognition_status = f"ERROR {type(exc).__name__}"
    return dict(is_spoof=is_spoof, is_pad_pending=is_pad_pending,
                pad_invalid_reason=track.pad_invalid_reason if pad_enabled_rt else None,
                recognition_called=submitted)


def resolve_pad_model_path(model_path):
    """A CLI model override must use the currently selected runtime contract."""
    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / "antispoof" / "models" / path
    path = path.resolve()
    if path != Path(config.PAD_MODEL_PATH).resolve():
        raise ValueError(
            "--pad-model khác model của cấu hình runtime hiện tại. "
            "Hãy chọn PAD_RUNTIME_CONFIG_PATH và model tương ứng trước khi chạy; "
            "không thể dùng threshold/preprocessing của model hiện tại cho model khác."
        )
    return path


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
        help=f"Đường dẫn model khớp cấu hình PAD runtime đã chọn (mặc định: {config.PAD_MODEL_PATH})"
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
    parser.add_argument('--capture-dir', help='Thư mục lưu crop PAD; nhấn C để chụp')
    parser.add_argument('--subject', default='unspecified', help='Mã người được chụp (không phải nhãn real/spoof)')
    parser.add_argument('--condition', default='unspecified', help='Điều kiện ánh sáng / camera của phiên')
    parser.add_argument('--label', choices=['real', 'fake'], default=None,
                        help='Gán nhãn ground-truth tự động cho phiên chụp (real hoặc fake)')
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    capture = None
    if args.capture_dir:
        from pad_capture import PADCapture
        capture = PADCapture(args.capture_dir, args.subject, args.condition, label=args.label)
        label_info = f" [nhãn tự động: '{args.label}']" if args.label else " [chưa gán nhãn]"
        print(f'[Capture] C: lưu crop vào {capture.directory}{label_info}')
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
    embedder = FaceEmbedder(config.MODEL_PACK_NAME, ctx_id=config.MODEL_CTX_ID,
                           intra_op_threads=config.RECOGNITION_INTRA_OP_THREADS)
    db = VectorDB(
        conninfo=config.DB_CONN_INFO,
        min_size=config.DB_POOL_MIN_SIZE,
        max_size=config.DB_POOL_MAX_SIZE,
        pool_timeout=config.RECOGNITION_DB_POOL_TIMEOUT_SECONDS,
        query_timeout_ms=config.RECOGNITION_DB_QUERY_TIMEOUT_MS,
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
            _pad_model_path = resolve_pad_model_path(args.pad_model)
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
            gamma_status = "ON (FORCED)" if config.FORCE_GAMMA else (
                "ON" if config.PAD_GAMMA_ENABLED else "OFF"
            )
            print(f"  PAD_GAMMA        = {gamma_status}  (target luma={config.PAD_GAMMA_TARGET:.0f})")
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
    if config.CAMERA_FORCE_RESOLUTION:
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
    resolution_request = (
        f"{config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}"
        if config.CAMERA_FORCE_RESOLUTION else "backend-default"
    )
    print(
        f"  CAMERA          = backend={camera_backend} "
        f"force_resolution={config.CAMERA_FORCE_RESOLUTION} "
        f"requested={resolution_request}@{config.CAMERA_FPS} "
        f"actual={actual_width}x{actual_height}@{actual_fps:.2f} "
        f"buffer={actual_buffer:.2f}"
    )
    print(f"  DETECTOR         = {_DETECTOR_BACKEND.upper()}")
    print(f"  MATCH_THRESHOLD  = {config.MATCH_THRESHOLD}")
    print(f"  RECOGNIZE_EVERY  = {config.RECOGNIZE_INTERVAL_SECONDS}s")
    print(f"  PAD_EVERY        = {config.PAD_INTERVAL_SECONDS}s")
    print(f"  PAD_CROP_SMOOTHING = {config.PAD_CROP_SMOOTHING}")
    print(f"  PAD_CROP_MODE = {config.PAD_CROP_MODE}")
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
            label = "recognition_submit" if name == "recognition" else name
            print(f"  {label}_mean_ms={mean_ms:.2f}")

    recognition_worker = RecognitionWorker(
        embedder, db, config.EMBEDDING_MODEL_VERSION, config.MATCH_THRESHOLD,
        queue_size=config.RECOGNITION_QUEUE_SIZE,
        cache_ttl=config.RECOGNITION_CACHE_TTL_SECONDS,
        cache_threshold=config.RECOGNITION_CACHE_THRESHOLD,
    )
    print(f"[Recognition worker] queue={config.RECOGNITION_QUEUE_SIZE} "
          f"ArcFace_threads={config.RECOGNITION_INTRA_OP_THREADS} "
          f"cache_ttl={config.RECOGNITION_CACHE_TTL_SECONDS}s "
          f"cache_threshold={recognition_worker.cache_threshold}")
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
            if not detector_due:
                tracker_t0 = time.perf_counter()
                tracked = tracker.update([], frame=frame, detector_called=False)
                counters["tracker_ms"] += (time.perf_counter() - tracker_t0) * 1000.0
                counters["tracker_updates"] += 1
            if detector_due or tracker.needs_redetection():
                detector_t0 = time.perf_counter()
                try:
                    detections = limit_faces(detector.detect(frame))
                except Exception as e:
                    print(f"[Detector error] {e}")
                    detections = []
                counters["detector_ms"] += (time.perf_counter() - detector_t0) * 1000.0
                counters["detector_calls"] += 1
                last_detection_time = now
                detection_called = True
                tracker_t0 = time.perf_counter()
                tracked = tracker.update(detections, frame=frame, detector_called=True)
                counters["tracker_ms"] += (time.perf_counter() - tracker_t0) * 1000.0
                counters["tracker_updates"] += 1

            selected = limit_faces([d for d, _ in tracked])
            selected_ids = {id(d) for d in selected}
            tracked = [(d, t) for d, t in tracked if id(d) in selected_ids]

            # ── PAD batch ────────────────────────────────────────────────
            if pad_enabled_rt and pad_predictor is not None and tracked:
                pad_t0 = time.perf_counter()
                counters["pad_calls"] += update_track_pad(
                    tracked, frame, pad_predictor, config.PAD_INTERVAL_SECONDS,
                    frame_id, detection_called, config.PAD_DIAGNOSTIC_LOG,
                    crop_smoothing=config.PAD_CROP_SMOOTHING,
                )
                counters["pad_ms"] += (time.perf_counter() - pad_t0) * 1000.0

            # Keep model input free of annotations from earlier tracks.
            display_frame = frame.copy()
            collect_recognition_results(recognition_worker, tracked, pad_enabled_rt)

            # ── Recognition + Attendance + Display ───────────────────────────
            for detection, track in tracked:
                bbox = detection["bbox"]
                recognition_t0 = time.perf_counter()

                orch_res = submit_recognition_step(
                    worker=recognition_worker,
                    track=track,
                    detection=detection,
                    frame=frame,
                    embedder=embedder,
                    pad_enabled_rt=pad_enabled_rt,
                )
                counters["recognition_ms"] += (time.perf_counter() - recognition_t0) * 1000.0
                if orch_res["recognition_called"]:
                    counters["recognition_calls"] += 1

                # Trạng thái hiển thị
                name, score, label_stale = track.display_result(config.RECOGNITION_LABEL_HOLD_SECONDS)
                is_pending = track.is_pending
                is_reverifying = (
                    not is_pending
                    and not orch_res["is_pad_pending"]
                    and not orch_res["is_spoof"]
                    and track.needs_recognition(config.RECOGNIZE_INTERVAL_SECONDS, pad_enabled=pad_enabled_rt)
                )

                # Disabled in Embed-only scope: Bỏ banner & log ghi attendance trực tiếp
                # if orch_res["attendance_attempted"]:
                #     attendance_msg_time = time.time()
                #     last_ts = orch_res["last_ts"]
                #     if orch_res["attendance_success"]:
                #         attendance_msg = f"{name}: {app_mode} SUCCESS"
                #         attendance_msg_color = (0, 255, 0)
                #         ts_str = f" lúc {last_ts.astimezone().strftime('%H:%M:%S')}" if last_ts else ""
                #         print(f"[ATTENDANCE] {attendance_msg}{ts_str}")
                #     else:
                #         reason = orch_res["attendance_reason"]
                #         attendance_msg = f"{name}: {reason}"
                #         attendance_msg_color = (0, 165, 255)
                #         ts_str = f" (Gần nhất: {last_ts.astimezone().strftime('%H:%M:%S')})" if last_ts else ""
                #         print(f"[ATTENDANCE BLOCKED] {name}: {reason}{ts_str}")

                draw_track(
                    display_frame, bbox, name, score,
                    is_pending=is_pending,
                    is_reverifying=is_reverifying or label_stale,
                    is_spoof=orch_res["is_spoof"],
                    is_pad_pending=orch_res["is_pad_pending"],
                    pad_invalid_reason=orch_res["pad_invalid_reason"],
                    recognition_error=track.recognition_error,
                    recognition_status=(
                        "LAST LABEL | " + track.recognition_status
                        if label_stale and name != "UNKNOWN" else track.recognition_status
                    ),
                    stable_label=True,
                )
                if pad_enabled_rt and pad_predictor is not None:
                    draw_pad_margin(display_frame, bbox, track, pad_predictor.logit_threshold)


            # ── Overlay FPS & PAD badge ───────────────────────────────────────
            if show_fps_rt:
                draw_fps(display_frame, avg_fps)
            draw_pad_badge(display_frame, pad_enabled_rt)
            
            # ── Draw mode & attendance message ────────────────────────────────
            cv2.putText(display_frame, f"MODE: {app_mode}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
            if attendance_msg and (time.time() - attendance_msg_time < 3.0):
                # Hiển thị thông báo (thành công hoặc cảnh báo) trong 3 giây
                cv2.putText(display_frame, attendance_msg, (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, attendance_msg_color, 2, cv2.LINE_AA)

            if capture is not None:
                lbl_tag = f" [{capture.label}]" if capture.label else ""
                capture_text = f"C: Save sample{lbl_tag} | Captured: {capture.saved_count}"
                cv2.putText(display_frame, capture_text, (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("SmartFace — Face Auth", display_frame)

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
            elif key == ord("c") and capture is not None:
                try:
                    paths = capture.save(frame, tracked, pad_predictor)
                    if paths:
                        print(f'[Capture] +{len(paths)} mat moi -> Tong so anh da chup: {capture.saved_count} ({capture.directory})')
                    else:
                        print('[Capture] Khong phat hien khuon mat de chup!')
                except Exception as exc:
                    print(f'[Capture error] {exc}')
            elif key == ord("p"):
                if pad_predictor is None:
                    try:
                        from antispoof import AntiSpoofPredictor
                        _pad_model_path = resolve_pad_model_path(args.pad_model)
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
                    for t in tracker.tracks:
                        t.invalidate_verification()
                    print(f"[PAD] {'ON' if pad_enabled_rt else 'OFF'}")
                else:
                    print("[PAD] Không thể bật — model PAD chưa sẵn sàng.")
            elif key == ord("f"):
                show_fps_rt = not show_fps_rt

    finally:
        print_runtime_summary()
        cap.release()
        cv2.destroyAllWindows()
        stopped = recognition_worker.close()
        print(f"[Recognition worker] {recognition_worker.stats} stopped={stopped}")


if __name__ == "__main__":
    main()
